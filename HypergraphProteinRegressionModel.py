# HypergraphProteinRegressionModel.py (全新实现)
import torch as th
from torch import nn
from dgl import function as fn


class SpectralHypergraphConvLayer(nn.Module):
    """
    谱聚类超图专用卷积层
    实现超边权重的自注意力机制和特征融合
    """

    def __init__(self, input_dim, hidden_dim, output_dim, bias=True, batchnorm=False, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 特征变换
        self.self_transform = nn.Linear(input_dim, output_dim, bias=bias)
        self.cluster_transform = nn.Linear(input_dim, output_dim, bias=bias)

        # 改进的多头超边注意力机制
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        self.hyperedge_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.head_dim, bias=bias),
                nn.ReLU(),
                nn.Linear(self.head_dim, 1, bias=bias),
                nn.Sigmoid()
            ) for _ in range(self.num_heads)
        ])

        # 超边注意力融合
        self.attn_fusion = nn.Linear(self.num_heads, 1, bias=False)

        # 残差连接投影层（当输入输出维度不一致时使用）
        self.residual_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=bias),
            nn.ReLU()
        ) if input_dim != output_dim else None

        # 特征融合注意力
        self.fusion_attention = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2, bias=bias),
            nn.Softmax(dim=-1)
        )

        # 正则化
        self.batchnorm = nn.BatchNorm1d(output_dim) if batchnorm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for transform in [self.self_transform, self.cluster_transform]:
            nn.init.xavier_uniform_(transform.weight)
            if transform.bias is not None:
                nn.init.zeros_(transform.bias)

        for m in [self.hyperedge_attention, self.fusion_attention]:
            for layer in m:
                if hasattr(layer, 'weight'):
                    nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # 初始化残差投影层
        if self.residual_proj:
            nn.init.xavier_uniform_(self.residual_proj[0].weight)
            nn.init.zeros_(self.residual_proj[0].bias)

    def forward(self, hypergraph, feat):
        """
        谱聚类超图的前向传播
        实现超边权重的动态学习和特征融合
        """
        device = feat.device
        n_proteins = feat.shape[0]
        n_hyperedges = hypergraph.num_nodes() - n_proteins
        input_feat = feat  #保存输入特征用于残差连接

        with hypergraph.local_scope():
            if n_proteins == 0 or hypergraph.num_nodes() < n_proteins + 1:
                return th.zeros(n_proteins, self.output_dim, device=device)

            # --------------------------
            # 步骤1: 蛋白质 → 超边传播（带节点注意力）
            # --------------------------
            # 初始化节点特征
            he_zeros = th.zeros(n_hyperedges, self.input_dim, device=device)
            hypergraph.ndata['h'] = th.cat([feat, he_zeros], dim=0)

            # 消息传递：蛋白质 → 超边
            hypergraph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'h_he'))
            he_feat_initial = hypergraph.ndata['h_he'][n_proteins:]

            # --------------------------
            # 步骤2: 多头超边注意力（动态学习超边权重）
            # --------------------------
            head_attn_weights = []
            for head in self.hyperedge_attention:
                head_attn = head(he_feat_initial)
                head_attn_weights.append(head_attn)

            # 融合多头注意力
            multi_head_weights = th.cat(head_attn_weights, dim=-1)
            he_attn_weights = self.attn_fusion(multi_head_weights)
            he_feat_weighted = he_feat_initial * he_attn_weights

            # --------------------------
            # 步骤3: 超边 → 蛋白质传播（带注意力权重）
            # --------------------------
            protein_zeros = th.zeros(n_proteins, self.input_dim, device=device)
            hypergraph.ndata['h'] = th.cat([protein_zeros, he_feat_weighted], dim=0)
            hypergraph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'h_p'))
            cluster_feat = hypergraph.ndata['h_p'][:n_proteins]

            # --------------------------
            # 步骤4: 特征变换和融合
            # --------------------------
            # 自身特征变换
            self_feat = self.self_transform(feat)

            # 簇特征变换
            cluster_feat_transformed = self.cluster_transform(cluster_feat)

            # 特征融合（使用注意力机制）
            features_concat = th.cat([self_feat, cluster_feat_transformed], dim=-1)
            fusion_weights = self.fusion_attention(features_concat)

            # 加权融合
            weight_self, weight_cluster = fusion_weights.unbind(dim=-1)
            fused_feat = (self_feat * weight_self.unsqueeze(-1) +
                          cluster_feat_transformed * weight_cluster.unsqueeze(-1))

            # --------------------------
            # 步骤5: 残差连接
            # --------------------------
            if self.residual_proj is not None:
                # 当输入输出维度不一致时，使用带激活的线性投影
                residual = self.residual_proj(input_feat)
            else:
                # 当输入输出维度一致时，直接使用输入特征
                residual = input_feat

            # 添加残差连接
            fused_feat = fused_feat + residual

            # --------------------------
            # 步骤6: 正则化
            # --------------------------
            if self.batchnorm:
                fused_feat = self.batchnorm(fused_feat)
            fused_feat = self.activation(fused_feat)
            if self.dropout:
                fused_feat = self.dropout(fused_feat)

            return fused_feat


class TopologicalHypergraphConvLayer(nn.Module):
    """
    拓扑超图卷积层（保持您原来的多阶邻居设计）
    """

    def __init__(self, input_dim, hidden_dim, output_dim, bias=True, batchnorm=False, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 各阶特征变换
        self.self_transform = nn.Linear(input_dim, output_dim, bias=bias)
        self.first_order_transform = nn.Linear(input_dim, output_dim, bias=bias)
        self.second_order_transform = nn.Linear(input_dim, output_dim, bias=bias)

        # 改进的多头超边注意力
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        self.hyperedge_attn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.head_dim, bias=bias),
                nn.ReLU(),
                nn.Linear(self.head_dim, 1, bias=bias),
                nn.Sigmoid()
            ) for _ in range(self.num_heads)
        ])

        # 超边注意力融合
        self.attn_fusion = nn.Linear(self.num_heads, 1, bias=False)

        # 通道注意力机制替换原来的特征融合注意力
        # 注意：这里我们针对的是3个特征源的注意力，而不是特征维度的注意力
        self.source_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(3, 3, 1, bias=False),
            nn.Sigmoid()
        )

        # 改进的残差连接投影层（当输入输出维度不一致时使用）
        self.residual_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=bias),
            nn.ReLU()
        ) if input_dim != output_dim else None

        # 正则化
        self.batchnorm = nn.BatchNorm1d(output_dim) if batchnorm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for transform in [self.self_transform, self.first_order_transform, self.second_order_transform]:
            nn.init.xavier_uniform_(transform.weight)
            if transform.bias is not None:
                nn.init.zeros_(transform.bias)

        for m in [self.hyperedge_attn]:
            for layer in m:
                if hasattr(layer, 'weight'):
                    nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # 初始化特征源注意力层
        for layer in self.source_attention:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # 初始化残差投影层
        if self.residual_proj:
            nn.init.xavier_uniform_(self.residual_proj[0].weight)
            nn.init.zeros_(self.residual_proj[0].bias)

    def _propagate_with_edge_type(self, hypergraph, feat, edge_type_range):
        """在指定类型的超边上进行传播"""
        device = feat.device
        n_proteins = feat.shape[0]
        n_hyperedges = hypergraph.num_nodes() - n_proteins

        # 创建掩码，只选择指定类型的边
        edge_mask = (hypergraph.edata['type'] >= edge_type_range[0]) & \
                    (hypergraph.edata['type'] < edge_type_range[1])

        if edge_mask.sum() == 0:
            return th.zeros_like(feat)

        # 创建子图进行传播
        subgraph = hypergraph.edge_subgraph(edge_mask.squeeze(), relabel_nodes=False)

        # 蛋白质 → 超边
        he_zeros = th.zeros(n_hyperedges, self.input_dim, device=device)
        subgraph.ndata['h'] = th.cat([feat, he_zeros], dim=0)
        subgraph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'h_he'))
        he_feat = subgraph.ndata['h_he'][n_proteins:]

        # 多头超边注意力
        head_attn_weights = []
        for head in self.hyperedge_attn:
            head_attn = head(he_feat)
            head_attn_weights.append(head_attn)

        # 融合多头注意力
        multi_head_weights = th.cat(head_attn_weights, dim=-1)
        he_attn = self.attn_fusion(multi_head_weights)
        he_weighted = he_feat * he_attn

        # 超边 → 蛋白质
        protein_zeros = th.zeros(n_proteins, self.input_dim, device=device)
        subgraph.ndata['h'] = th.cat([protein_zeros, he_weighted], dim=0)
        subgraph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'h_p'))
        feat_agg = subgraph.ndata['h_p'][:n_proteins]

        return feat_agg

    def forward(self, hypergraph, feat):
        device = feat.device
        n_proteins = feat.shape[0]
        input_feat = feat  # 保存输入特征用于残差连接

        with hypergraph.local_scope():
            if n_proteins == 0 or hypergraph.num_nodes() < n_proteins + 1:
                return th.zeros(n_proins, self.output_dim, device=device)

            # 获取超边类型范围 ==============================================
            if not hasattr(hypergraph, '_hyperedge_types'):
                # 如果没有类型信息，使用简单传播
                print("No hyperedge types found. Using simple propagation.")
                return self.self_transform(feat)

            type_ranges = hypergraph._hyperedge_types

            # 0阶特征：自身特征
            feat_0 = self.self_transform(feat)

            # 1阶特征：在1阶超边上传播
            feat_1_agg = self._propagate_with_edge_type(hypergraph, feat, type_ranges['1st_range'])
            feat_1 = self.first_order_transform(feat_1_agg)

            # 2阶特征：在2阶超边上传播
            feat_2_agg = self._propagate_with_edge_type(hypergraph, feat, type_ranges['2nd_range'])
            feat_2 = self.second_order_transform(feat_2_agg)

            # 使用通道注意力机制进行特征融合
            features_by_order = [feat_0, feat_1, feat_2]
            stacked_features = th.stack(features_by_order, dim=-1)  # [batch, features, 3]
            
            # 应用特征源注意力
            # 需要调整维度以适应Conv1d [batch, channels, length]
            attention_weights = self.source_attention(stacked_features.transpose(1, 2)).transpose(1, 2)
            weighted_features = stacked_features * attention_weights
            
            # 聚合特征
            fused_feat = th.sum(weighted_features, dim=-1)

            # --------------------------
            # 残差连接
            # --------------------------
            if self.residual_proj is not None:
                # 当输入输出维度不一致时，使用带激活的线性投影
                residual = self.residual_proj(input_feat)
            else:
                # 当输入输出维度一致时，直接使用输入特征
                residual = input_feat

            # 添加残差连接
            fused_feat = fused_feat + residual

            # 正则化
            if self.batchnorm:
                fused_feat = self.batchnorm(fused_feat)
            fused_feat = self.activation(fused_feat)
            if self.dropout:
                fused_feat = self.dropout(fused_feat)

            return fused_feat


class DualAttentionFusionLayer(nn.Module):
    """
    双重注意力融合层
    对KNN拓扑特征和谱聚类特征进行注意力融合
    """

    def __init__(self, in_feats, attn_hid_feats, dropout=0.0):
        super(DualAttentionFusionLayer, self).__init__()
        self.in_feats = in_feats

        # 改进的第一层注意力：特征重要性（使用更深的网络）
        self.feature_attention = nn.Sequential(
            nn.Linear(in_feats * 2, attn_hid_feats),
            nn.ReLU(),
            nn.Linear(attn_hid_feats, attn_hid_feats // 2),
            nn.ReLU(),
            nn.Linear(attn_hid_feats // 2, 2),
            nn.Softmax(dim=-1)
        )

        # 改进的第二层注意力：跨通道注意力（使用门控机制）
        self.channel_attention = nn.Sequential(
            nn.Linear(in_feats, attn_hid_feats // 4),
            nn.ReLU(),
            nn.Linear(attn_hid_feats // 4, in_feats),
            nn.Sigmoid()
        )

        # 添加门控机制
        self.gate = nn.Sequential(
            nn.Linear(in_feats * 2, in_feats),
            nn.ReLU(),
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, feat_knn, feat_spectral):
        # 拼接两种特征
        concatenated_features = th.cat([feat_knn, feat_spectral], dim=-1)

        # 计算特征融合权重
        fusion_weights = self.feature_attention(concatenated_features)
        weight_knn, weight_spectral = fusion_weights.unbind(dim=-1)

        # 应用通道注意力
        channel_attn_knn = self.channel_attention(feat_knn)
        channel_attn_spectral = self.channel_attention(feat_spectral)

        # 双重注意力融合
        attended_knn = feat_knn * channel_attn_knn
        attended_spectral = feat_spectral * channel_attn_spectral

        # 使用门控机制进一步优化融合
        gate_input = th.cat([attended_knn, attended_spectral], dim=-1)
        gate_weight = self.gate(gate_input)

        # 加权融合，通过门控机制动态控制两种特征的融合比例
        fused_feat = (attended_knn * weight_knn.unsqueeze(-1) * gate_weight +
                      attended_spectral * weight_spectral.unsqueeze(-1) * (1 - gate_weight))

        if self.dropout:
            fused_feat = self.dropout(fused_feat)

        return fused_feat, fusion_weights


class HypergraphProteinRegressionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # KNN拓扑超图嵌入层
        self.hyper_emb_knn = TopologicalHypergraphConvLayer(
            input_dim=args.pssm_dim,
            hidden_dim=args.hid_feats,
            output_dim=args.out_feats,
            bias=args.bias,
            batchnorm=args.batchnorm,
            dropout=args.dropout
        )

        # 谱聚类超图嵌入层
        self.hyper_emb_spectral = SpectralHypergraphConvLayer(
            input_dim=args.pssm_dim,
            hidden_dim=args.hid_feats,
            output_dim=args.out_feats,
            bias=args.bias,
            batchnorm=args.batchnorm,
            dropout=args.dropout
        )

        # 双重注意力特征融合层
        self.dual_fusion = DualAttentionFusionLayer(
            in_feats=args.out_feats,
            attn_hid_feats=args.hid_feats // 2,
            dropout=args.dropout
        )

        # 动态构建回归器（支持自定义层数和维度）
        regressor_dims = [args.out_feats] + getattr(args, 'regressor_layers', [args.hid_feats, 256, 128, 64]) + [1]
        regressor_layers = []
        
        for i in range(len(regressor_dims) - 1):
            regressor_layers.append(nn.Linear(regressor_dims[i], regressor_dims[i+1]))
            if i < len(regressor_dims) - 2:  # 不在最后一层添加激活函数
                if getattr(args, 'batchnorm', True):
                    regressor_layers.append(nn.BatchNorm1d(regressor_dims[i+1]))
                regressor_layers.append(nn.ReLU())
                if getattr(args, 'dropout', 0) > 0:
                    regressor_layers.append(nn.Dropout(args.dropout))
        
        self.regressor = nn.Sequential(*regressor_layers)

        # 改进的回归器残差连接投影层（使用更深的网络）
        # 根据中间层维度确定残差连接的结构
        intermediate_dim = regressor_dims[1] if len(regressor_dims) > 2 else 1
        self.regressor_residual_proj = nn.Sequential(
            nn.Linear(args.out_feats, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1)
        )

    def forward(self, hypergraph_knn, hypergraph_spectral, protein_feat):
        # 输入检查
        assert protein_feat.shape[1] == self.args.pssm_dim, \
            f"Input feature dimension mismatch: {protein_feat.shape[1]} vs {self.args.pssm_dim}"

        # KNN拓扑特征提取
        feat_knn = self.hyper_emb_knn(hypergraph_knn, protein_feat)

        # 谱聚类特征提取
        feat_spectral = self.hyper_emb_spectral(hypergraph_spectral, protein_feat)

        # 中间特征检查
        assert feat_knn.shape == feat_spectral.shape, \
            f"Feature shape mismatch: KNN {feat_knn.shape} vs Spectral {feat_spectral.shape}"

        # 双重注意力融合
        hyper_feat, fusion_weights = self.dual_fusion(feat_knn, feat_spectral)

        # 回归预测 with 残差连接
        regressor_output = self.regressor(hyper_feat)
        residual = self.regressor_residual_proj(hyper_feat)
        pred = regressor_output + residual

        return pred.squeeze(-1)
