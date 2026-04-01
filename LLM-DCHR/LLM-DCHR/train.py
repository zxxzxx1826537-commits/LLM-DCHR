# train.py (纯粹全图训练完整版 - 修复超图构建逻辑)
import torch as th
import torch.nn as nn
import numpy as np
import os
import argparse
import dgl
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold, train_test_split
from HypergraphProteinRegressionModel import HypergraphProteinRegressionModel
from utils import (load_protein_pssm, load_similarity_network, fuse_similarity_networks,
                   load_protein_labels, filter_adjacency_matrix)

# 避免多线程库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加多进程保护，避免重复初始化
if __name__ != '__main__':
    # 禁用不必要的日志输出
    import logging
    logging.disable(logging.CRITICAL)
else:
    # 只在主进程中导入和执行需要的代码
    pass

# ==============================================================================
# 1. 环境检查与配置
# ==============================================================================
try:
    import dgl

    print(f"DGL version: {dgl.__version__}")
except ImportError as e:
    print(f"Failed to import DGL: {e}")
    dgl = None

try:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
except:
    device = th.device("cpu")
    print("Using CPU device")


# --- Args 类 ---
class Args:
    def __init__(self, **kwargs):
        # 请根据您的实际环境修改以下路径！
        self.pssm_dir = kwargs.get('pssm_dir', r"D:\zxx\protT5_lysine_window")
        self.cos_net_path = kwargs.get('cos_net_path', r"D:\zxx\Construct Protein Graph_Matrix\Gaussian_kernel_similarity_net\similarity_net.csv")
        self.lev_net_path = kwargs.get('lev_net_path', r"D:\zxx\Construct Protein Graph_Matrix\Levenshtein_similarity_net\similarity_net.csv")
        self.dom_net_path = kwargs.get('dom_net_path', r"D:\zxx\Construct Protein Graph_Matrix\Domain_similarity_net\similarity_net1.csv")
        self.label_path = kwargs.get('label_path', r"D:\zxx\protein_labels.csv")

        # 可调参数（设置默认值或从 kwargs 获取）
        self.lambda_param = kwargs.get('lambda_param', 0.2)
        self.pssm_dim = kwargs.get('pssm_dim', 1024)
        self.hid_feats = kwargs.get('hid_feats', 256)
        self.out_feats = kwargs.get('out_feats', 1024)
        self.bias = kwargs.get('bias', True)
        self.batchnorm = kwargs.get('batchnorm', True)
        self.dropout = kwargs.get('dropout', 0.4)
        self.lr = kwargs.get('lr', 0.0003)
        self.wd = kwargs.get('wd', 0.001)
        self.k= kwargs.get('k', 32)
        self.n_clusters_spectral = kwargs.get('n_clusters_spectral', 166)
        self.epochs = kwargs.get('epochs', 2000)
        
        # 回归头参数
        self.regressor_layers = kwargs.get('regressor_layers', [256,256, 32, 32])

        # 交叉验证参数
        self.n_splits = kwargs.get('n_splits', 10)  # 五折交叉验证

        # 设备设置
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

args = Args(lambda_param=0.2,hid_feats=256,dropout=0.4,lr=0.0003,wd=0.001,k=32,n_clusters_spectral=166,regressor_layers=[256,256, 32, 32], n_splits=10)
#args = Args(lambda_param=0.10504873720387847,hid_feats=256,dropout=0.05066515165687713, lr=1.1576339615996762e-05, wd=2.1094372562209593e-06, epochs=1265, regressor_layers=[512, 64, 32], n_splits=10)
#args = Args(lambda_param=0.3,hid_feats=256,dropout=0.4,lr=0.0003,wd=0.001,k=11,n_clusters_spectral=166,regressor_layers=[256,256, 32, 32], n_splits=10)
#args = Args(lambda_param=0.3,hid_feats=512,dropout=0.5,lr=0.0003,wd=0.001,k=21,n_clusters_spectral=105,regressor_layers=[128, 64, 32, 64, 32], n_splits=10)
def print_model_info(model):
    """打印模型的参数信息"""
    try:
        print("\n" + "="*50)
        print("MODEL INFO START")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("Model Information")
        print("="*50)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Hidden Dimension: {args.hid_feats}")
        print(f"Output Dimension: {args.out_feats}")
        print("="*50)
        print("MODEL INFO END")
        
        return total_params
    except Exception as e:
        print(f"Error in print_model_info: {e}")
        import traceback
        traceback.print_exc()
        return 0

# ==============================================================================
# 2. Hypergraph Builder Classes (修改：使用谱聚类)
# ==============================================================================
class ProteinHypergraphBuilderSpectral:
    def __init__(self, n_clusters=100):
        self.n_clusters = n_clusters


    def build_hypergraph_from_similarity(self, similarity_matrix, protein2idx,):
        """
        基于蛋白质相似矩阵构建谱聚类超图
        """
        # 确保输入是numpy数组而不是matrix
        similarity_matrix = np.asarray(similarity_matrix)
        
        n_proteins = similarity_matrix.shape[0]
        if n_proteins != len(protein2idx):
            raise ValueError(f"相似矩阵有{n_proteins}个蛋白质，但protein2idx有{len(protein2idx)}个蛋白质")

        n_clusters = min(self.n_clusters, n_proteins)
        print(f"谱聚类: {n_proteins}个蛋白质 -> {n_clusters}个簇")

        # 使用谱聚类
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42,
            n_init=10,
            assign_labels='kmeans'
        )

        labels = spectral.fit_predict(similarity_matrix)

        # 构建超边
        hyperedges = [[] for _ in range(n_clusters)]
        for i in range(n_proteins):
            hyperedges[labels[i]].append(i)

        valid_hyperedges = [h for h in hyperedges if len(h) >= 1]
        n_hyperedges = len(valid_hyperedges)
        print(f"谱聚类超图构建完成: {n_hyperedges}个有效超边")

        # 基于相似度计算超边权重
        hyperedge_weights = self._calculate_hyperedge_weights_from_similarity(valid_hyperedges, similarity_matrix)

        return self._hyperedges_to_dgl(valid_hyperedges, hyperedge_weights, n_proteins)

    def _calculate_hyperedge_weights_from_similarity(self, hyperedges, similarity_matrix):
        """基于超边内节点的相似度计算权重"""
        weights = []

        for nodes in hyperedges:
            if len(nodes) <= 1:
                weights.append(1.0)
                continue

            # 计算超边内节点的平均相似度
            total_similarity = 0
            pair_count = 0

            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node_i, node_j = nodes[i], nodes[j]
                    total_similarity += similarity_matrix[node_i, node_j]
                    pair_count += 1

            if pair_count > 0:
                avg_similarity = total_similarity / pair_count
                weight = 0.5 + 1.5 * avg_similarity  # 映射到 [0.5, 2.0]
            else:
                weight = 1.0

            weights.append(weight)

        return weights

    def _hyperedges_to_dgl(self, hyperedges, hyperedge_weights, n_nodes):
        """构建DGL图（谱聚类超图不需要类型标签）"""
        try:
            src_nodes = []
            dst_nodes = []
            edge_weights = []

            for he_idx, nodes in enumerate(hyperedges):
                hyperedge_node_id = n_nodes + he_idx
                weight = hyperedge_weights[he_idx]

                # 蛋白质节点 -> 超边节点
                for protein_idx in nodes:
                    src_nodes.append(protein_idx)
                    dst_nodes.append(hyperedge_node_id)
                    edge_weights.append(weight)

                # 超边节点 -> 蛋白质节点（双向）
                for protein_idx in nodes:
                    src_nodes.append(hyperedge_node_id)
                    dst_nodes.append(protein_idx)
                    edge_weights.append(weight)

            if not src_nodes:
                hypergraph = dgl.graph(([0], [0]))
                hypergraph.edata['e'] = th.tensor([1.0], dtype=th.float32).reshape(-1, 1)
                return hypergraph

            hypergraph = dgl.graph((th.tensor(src_nodes), th.tensor(dst_nodes)))
            hypergraph.edata['e'] = th.tensor(edge_weights, dtype=th.float32).reshape(-1, 1)

            print(f"谱聚类超图构建完成:")
            print(f"  超边数量: {len(hyperedges)}")
            print(f"  总节点: {hypergraph.num_nodes()}")
            print(f"  总边: {hypergraph.num_edges()}")

            return hypergraph

        except Exception as e:
            print(f"Error creating DGL graph: {e}")
            hypergraph = dgl.graph(([0], [0]))
            hypergraph.edata['e'] = th.tensor([1.0], dtype=th.float32).reshape(-1, 1)
            return hypergraph


class ProteinHypergraphBuilder:
    def __init__(self, top_k=20):
        # self.top_k = top_k
        self.k_first_order= top_k
        self.k_second_order= top_k


    def build_hypergraph_from_adj(self, adj_matrix, protein2idx):
        """基于邻接矩阵构建蛋白质超图（优化版本）"""
        n_proteins_from_adj = adj_matrix.shape[0]
        n_proteins_from_idx = len(protein2idx)

        if n_proteins_from_adj != n_proteins_from_idx:
            raise ValueError(f"严重错误: 邻接矩阵有{n_proteins_from_adj}个蛋白质, "
                             f"但protein2idx有{n_proteins_from_idx}个蛋白质。")

        print(f"✓ 维度验证通过: 所有数据源都包含{n_proteins_from_adj}个蛋白质")
        n_proteins = n_proteins_from_adj

        # 分别存储两种类型的超边
        hyperedges_1st = []  # 1阶超边: 自身 + 直接邻居
        hyperedges_2nd = []  # 2阶超边: 自身 + 二阶邻居

        hyperedge_weights_1st = []
        hyperedge_weights_2nd = []

        # 确保adj_matrix是numpy数组而不是矩阵
        adj_matrix = np.asarray(adj_matrix)

        # 创建二值化邻接矩阵用于邻居查找
        adj_bin = (adj_matrix > 0).astype(np.int32)
        np.fill_diagonal(adj_bin, 0)  # 移除自环

        print("使用向量化方式构建超图...")
        
        # 1阶超边: 自身 + 直接邻居
        print("构建1阶超边...")
        first_order_neighbors = self._get_first_order_neighbors_vectorized(adj_matrix)
        hyperedges_1st = []
        for center_idx in range(n_proteins):
            hyperedge_1st = [center_idx] + first_order_neighbors[center_idx].tolist()
            hyperedges_1st.append(list(set(hyperedge_1st)))

        # 2阶超边: 自身 + 二阶邻居（去除自身和一阶邻居）
        print("构建2阶超边...")
        second_order_neighbors = self._get_second_order_neighbors_vectorized(adj_matrix, first_order_neighbors)
        hyperedges_2nd = []
        for center_idx in range(n_proteins):
            hyperedge_2nd = second_order_neighbors[center_idx]
            hyperedges_2nd.append(list(set(hyperedge_2nd)))

        # 计算各种超边的权重
        print("计算超边权重...")
        hyperedge_weights_1st = self._calculate_hyperedge_weights_vectorized(hyperedges_1st, adj_matrix)
        hyperedge_weights_2nd = self._calculate_hyperedge_weights_vectorized(hyperedges_2nd, adj_matrix)

        # 构建包含两种类型超边的DGL图
        return self._multi_type_hyperedges_to_dgl(
            hyperedges_1st, hyperedges_2nd,
            hyperedge_weights_1st, hyperedge_weights_2nd,
            n_proteins
        )

    def _calculate_hyperedge_weights(self, hyperedges, adj_matrix):
        """计算超边权重"""
        weights = []
        for nodes in hyperedges:
            if len(nodes) <= 1:
                weights.append(1.0)
                continue

            total_strength = 0
            pair_count = 0

            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node_i, node_j = nodes[i], nodes[j]
                    total_strength += adj_matrix[node_i, node_j]
                    pair_count += 1

            if pair_count > 0:
                avg_strength = total_strength / pair_count
                weight = 0.5 + 1.5 * avg_strength  # 映到 [0.5, 2.0]
            else:
                weight = 1.0

            weights.append(weight)

        return weights

    def _calculate_hyperedge_weights_vectorized(self, hyperedges, adj_matrix):
        """向量化计算超边权重"""
        weights = []
        for nodes in hyperedges:
            if len(nodes) <= 1:
                weights.append(1.0)
                continue

            # 使用向量化操作计算所有节点对之间的相似度
            node_array = np.array(nodes)
            # 构建节点对索引
            i_indices, j_indices = np.meshgrid(node_array, node_array, indexing='ij')
            # 获取所有节点对的相似度
            similarities = adj_matrix[i_indices, j_indices]
            # 只取上三角部分（排除对角线）
            mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
            valid_sims = similarities[mask]
            
            if len(valid_sims) > 0:
                avg_similarity = np.mean(valid_sims)
                weight = 0.5 + 1.5 * avg_similarity  # 映射到 [0.5, 2.0]
            else:
                weight = 1.0

            weights.append(weight)

        return weights

    def _multi_type_hyperedges_to_dgl(self, hyperedges_1st, hyperedges_2nd,
                                      weights_1st, weights_2nd, n_nodes):
        """构建包含两种类型超边的DGL图"""
        try:
            src_nodes = []
            dst_nodes = []
            edge_weights = []
            edge_types = []  # 记录边类型

            # 辅助函数：添加超边到图
            def add_hyperedges(hyperedges, weights, edge_type_offset, edge_type_label):
                nonlocal src_nodes, dst_nodes, edge_weights, edge_types
                for he_idx, nodes in enumerate(hyperedges):
                    hyperedge_node_id = n_nodes + he_idx + edge_type_offset
                    weight = weights[he_idx]

                    # 蛋白质节点 -> 超边节点
                    for protein_idx in nodes:
                        src_nodes.append(protein_idx)
                        dst_nodes.append(hyperedge_node_id)
                        edge_weights.append(weight)
                        edge_types.append(edge_type_label)  # 记录类型

                    # 超边节点 -> 蛋白质节点（双向）
                    for protein_idx in nodes:
                        src_nodes.append(hyperedge_node_id)
                        dst_nodes.append(protein_idx)
                        edge_weights.append(weight)
                        edge_types.append(edge_type_label)  # 记录类型

            # 添加两种类型的超边，每种类型分配不同的ID范围和标签
            add_hyperedges(hyperedges_1st, weights_1st, 0, 1)  # 1阶超边: 类型1
            add_hyperedges(hyperedges_2nd, weights_2nd, len(hyperedges_1st), 2)  # 2阶超边: 类型2

            if not src_nodes:
                hypergraph = dgl.graph(([0], [0]))
                hypergraph.edata['e'] = th.tensor([1.0], dtype=th.float32).reshape(-1, 1)
                hypergraph.edata['type'] = th.tensor([1], dtype=th.long).reshape(-1, 1)
                return hypergraph

            hypergraph = dgl.graph((th.tensor(src_nodes), th.tensor(dst_nodes)))
            hypergraph.edata['e'] = th.tensor(edge_weights, dtype=th.float32).reshape(-1, 1)
            hypergraph.edata['type'] = th.tensor(edge_types, dtype=th.long).reshape(-1, 1)

            # 存储超边类型信息供卷积层使用
            hypergraph._hyperedge_types = {
                '1st_range': (0, len(hyperedges_1st)),
                '2nd_range': (len(hyperedges_1st), len(hyperedges_1st) + len(hyperedges_2nd)),
                'type_labels': {
                    '1st': 1,
                    '2nd': 2
                }
            }

            print(f"超图构建完成:")
            print(f"  1阶超边: {len(hyperedges_1st)}")
            print(f"  2阶超边: {len(hyperedges_2nd)}")
            print(f"  总超边: {len(hyperedges_1st) + len(hyperedges_2nd)}")
            print(f"  总节点: {hypergraph.num_nodes()}")
            print(f"  总边: {hypergraph.num_edges()}")

            return hypergraph

        except Exception as e:
            print(f"Error creating DGL graph: {e}")
            hypergraph = dgl.graph(([0], [0]))
            hypergraph.edata['e'] = th.tensor([1.0], dtype=th.float32).reshape(-1, 1)
            hypergraph.edata['type'] = th.tensor([1], dtype=th.long).reshape(-1, 1)
            return hypergraph

    def _get_first_order_neighbors_vectorized(self, similarity_matrix):
        """向量化获取一阶邻居"""
        n_proteins = similarity_matrix.shape[0]
        first_order_neighbors = []

        # 向量化处理所有节点
        for center_idx in tqdm(range(n_proteins), desc="获取一阶邻居"):
            first_order_similarities = similarity_matrix[center_idx].copy()
            first_order_similarities[center_idx] = 0  # 确保不包含自身

            # 获取top-k一阶邻居
            positive_mask = first_order_similarities > 0
            positive_indices = np.where(positive_mask)[0]
            positive_count = len(positive_indices)

            if positive_count <= self.k_first_order:
                first_order_indices = positive_indices
            else:
                # 获取相似度最高的k个一阶邻居
                top_similarities = first_order_similarities[positive_indices]
                topk_indices = np.argpartition(top_similarities, -self.k_first_order)[-self.k_first_order:]
                selected_positive_indices = positive_indices[topk_indices]
                # 按相似度排序
                sorted_indices = selected_positive_indices[
                    np.argsort(first_order_similarities[selected_positive_indices])[::-1]
                ]
                first_order_indices = sorted_indices

            first_order_neighbors.append(first_order_indices)

        return first_order_neighbors

    def _get_second_order_neighbors_vectorized(self, similarity_matrix, first_order_neighbors):
        """向量化获取二阶邻居"""
        n_proteins = similarity_matrix.shape[0]
        second_order_neighbors = []

        if self.k_second_order <= 0:
            return [[] for _ in range(n_proteins)]

        # 预计算所有一阶邻居的集合用于快速查找
        first_order_sets = [set(neighbors) for neighbors in first_order_neighbors]

        for center_idx in tqdm(range(n_proteins), desc="获取二阶邻居"):
            first_order_indices = first_order_neighbors[center_idx]
            first_order_set = first_order_sets[center_idx]

            # 收集所有一阶邻居的邻居
            all_second_neighbors = {}
            for first_neighbor in first_order_indices:
                # 获取一阶邻居的邻居
                neighbor_neighbors = np.where(similarity_matrix[first_neighbor] > 0)[0]
                # 排除中心节点和已经在一阶邻居中的节点
                for second_neighbor in neighbor_neighbors:
                    if second_neighbor != center_idx and second_neighbor not in first_order_set:
                        if second_neighbor not in all_second_neighbors:
                            all_second_neighbors[second_neighbor] = 0
                        #累计连接强度
                        all_second_neighbors[second_neighbor] += (similarity_matrix[first_neighbor, second_neighbor] + similarity_matrix[center_idx, first_neighbor])

            second_order_items = list(all_second_neighbors.items())
            second_order_indices = [item[0] for item in second_order_items]

            # 如果二阶邻居数量超过限制，选取最重要的
            if len(second_order_indices) > self.k_second_order:
                # 获取连接强度
                strengths = [item[1] for item in second_order_items]
                strengths = np.array(strengths)

                # 获取top-k二阶邻居
                topk_indices = np.argpartition(strengths, -self.k_second_order)[-self.k_second_order:]
                selected_second_order = [second_order_indices[i] for i in topk_indices]
                # 按强度排序
                selected_second_order = [selected_second_order[i] for i in
                                         np.argsort([strengths[i] for i in topk_indices])[::-1]]
            else:
                selected_second_order = second_order_indices

            second_order_neighbors.append(selected_second_order)

        return second_order_neighbors
# ==============================================================================
# 3. 核心训练与测试函数 (Pure Full-Batch)
# ==============================================================================

def train(model, all_feats, all_labels, train_idx, val_idx, criterion, optimizer, args, hypergraph_knn,
          hypergraph_spectral):
    best_val_rmse = np.inf
    patience = 100  # 早停耐心值
    patience_counter = 0  # 早停计数器

    all_feats_device = all_feats.to(args.device)
    all_labels_device = all_labels.to(args.device)
    hypergraph_knn_device = hypergraph_knn.to(args.device)
    hypergraph_spectral_device = hypergraph_spectral.to(args.device)

    train_idx_th = th.tensor(train_idx, device=args.device)
    val_idx_th = th.tensor(val_idx, device=args.device)

    model.to(args.device)
    # 梯度裁剪
    #grad_clip_value = 1.0
    print(f"Starting pure Full-Batch training on {args.device} for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # --- 训练阶段 ---
        model.train()
        optimizer.zero_grad()

        all_pred = model(hypergraph_knn_device, hypergraph_spectral_device, all_feats_device)

        train_pred = all_pred[train_idx_th].squeeze()
        train_label = all_labels_device[train_idx_th].squeeze()

        if train_pred.dim() == 0: train_pred = train_pred.unsqueeze(0)

        loss = criterion(train_pred, train_label)

        loss.backward()

        # 梯度裁剪
        #th.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        optimizer.step()
        train_loss = loss.item()

        # --- 验证阶段 ---
        model.eval()
        with th.no_grad():
            val_pred = all_pred[val_idx_th].squeeze()
            val_label = all_labels_device[val_idx_th].squeeze()

            if val_pred.dim() == 0: val_pred = val_pred.unsqueeze(0)

            val_loss = criterion(val_pred, val_label).item()

            val_preds_np = val_pred.cpu().numpy()
            val_labels_np = val_label.cpu().numpy()
            val_mse = np.mean((val_preds_np - val_labels_np) ** 2)
            val_rmse = np.sqrt(val_mse)
            val_mae = np.mean(np.abs(val_preds_np - val_labels_np))
            total_variance = np.sum((val_labels_np - np.mean(val_labels_np)) ** 2)
            val_r2 = 1 - (np.sum((val_labels_np - val_preds_np) ** 2) / total_variance) if total_variance != 0 else 1.0

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0  # 重置早停计数器
            th.save(model.state_dict(), "best_hypergraph_protein_model.pth")
            print(f"Epoch {epoch + 1}: Model saved. Val RMSE: {best_val_rmse:.4f}")
        else:
            patience_counter += 1  # 增加早停计数器

        print(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MSE: {val_mse:.4f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f} | Val R²: {val_r2:.4f}")
        
        # 早停机制
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return best_val_rmse


def test(model, all_feats, all_labels, test_idx, criterion, args, hypergraph_knn, hypergraph_spectral):
    model.load_state_dict(th.load("best_hypergraph_protein_model.pth"))
    model.eval()

    all_feats_device = all_feats.to(args.device)
    all_labels_device = all_labels.to(args.device)
    hypergraph_knn_device = hypergraph_knn.to(args.device)
    hypergraph_spectral_device = hypergraph_spectral.to(args.device)

    test_idx_th = th.tensor(test_idx, device=args.device)

    with th.no_grad():
        all_pred = model(hypergraph_knn_device, hypergraph_spectral_device, all_feats_device)

        test_pred = all_pred[test_idx_th].squeeze()
        test_label = all_labels_device[test_idx_th].squeeze()

        if test_pred.dim() == 0: test_pred = test_pred.unsqueeze(0)

        test_loss = criterion(test_pred, test_label).item()

        test_preds_np = test_pred.cpu().numpy()
        test_labels_np = test_label.cpu().numpy()
        test_mse = np.mean((test_preds_np - test_labels_np) ** 2)
        test_rmse = np.sqrt(test_mse)
        test_mae = np.mean(np.abs(test_preds_np - test_labels_np))
        total_variance = np.sum((test_labels_np - np.mean(test_labels_np)) ** 2)
        test_r2 = 1 - (np.sum((test_labels_np - test_preds_np) ** 2) / total_variance) if total_variance != 0 else 1.0

    print("=" * 50)
    print("Test Results (Best Model)")
    print(f"Test Loss: {test_loss:.4f} | Test MSE: {test_mse:.4f} | Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f} | Test R²: {test_r2:.4f}")
    print("=" * 50)
    return test_preds_np, test_mse, test_mae, test_r2, test_rmse


def cross_validate(model_class, all_feats, all_labels, indices, criterion, args, hypergraph_knn, hypergraph_spectral):
    """执行五折交叉验证"""
    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(indices)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{args.n_splits}")
        print(f"{'='*60}")
        
        # 进一步将训练验证集分为训练集和验证集 (80% 训练, 20% 验证)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42)
        
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")
        
        # 为每一折创建新的模型实例
        model = model_class(args).to(args.device)
        
        # 打印模型信息（仅在第一折打印）
        if fold == 0:
            print_model_info(model)
        
        optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        # 训练模型
        best_val_rmse = train(model, all_feats, all_labels, train_idx, val_idx, criterion, optimizer, args, 
                             hypergraph_knn, hypergraph_spectral)
        
        # 测试模型
        test_preds, test_mse, test_mae, test_r2, test_rmse = test(model, all_feats, all_labels, test_idx, criterion, args,
                                                       hypergraph_knn, hypergraph_spectral)
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_rmse': best_val_rmse,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_rmse': test_rmse
        })
        
        print(f"Fold {fold + 1} Results:")
        print(f"  Best Val RMSE: {best_val_rmse:.4f}")
        print(f"  Test MSE: {test_mse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
    
    # 计算平均结果
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results (Average of {args.n_splits} folds)")
    print(f"{'='*60}")
    
    avg_val_rmse = np.mean([r['best_val_rmse'] for r in fold_results])
    avg_test_mse = np.mean([r['test_mse'] for r in fold_results])
    avg_test_rmse = np.mean([r['test_rmse'] for r in fold_results])
    avg_test_mae = np.mean([r['test_mae'] for r in fold_results])
    avg_test_r2 = np.mean([r['test_r2'] for r in fold_results])
    
    std_val_rmse = np.std([r['best_val_rmse'] for r in fold_results])
    std_test_mse = np.std([r['test_mse'] for r in fold_results])
    std_test_rmse = np.std([r['test_rmse'] for r in fold_results])
    std_test_mae = np.std([r['test_mae'] for r in fold_results])
    std_test_r2 = np.std([r['test_r2'] for r in fold_results])
    
    print(f"Average Best Val RMSE: {avg_val_rmse:.4f} ± {std_val_rmse:.4f}")
    print(f"Average Test MSE: {avg_test_mse:.4f} ± {std_test_mse:.4f}")
    print(f"Average Test RMSE: {avg_test_rmse:.4f} ± {std_test_rmse:.4f}")
    print(f"Average Test MAE: {avg_test_mae:.4f} ± {std_test_mae:.4f}")
    print(f"Average Test R²: {avg_test_r2:.4f} ± {std_test_r2:.4f}")
    
    return fold_results


# ==============================================================================
# 4. 主程序执行块 (完整数据加载与超图创建)
# ==============================================================================

def main():
    print("\n--- 1. Data Loading ---")
    pssm_dict = load_protein_pssm(args.pssm_dir)
    cos_proteins, cos2idx, cos_adj = load_similarity_network(args.cos_net_path)
    lev_proteins, lev2idx, lev_adj = load_similarity_network(args.lev_net_path)
    dom_proteins, dom2idx, dom_adj = load_similarity_network(args.dom_net_path)

    print("\n--- 2. Data Filtering ---")
    common_proteins = set(pssm_dict.keys()) & set(cos_proteins) & set(lev_proteins) & set(dom_proteins)
    print(f"Found {len(common_proteins)} common proteins across all data sources")

    filtered_pssm_dict = {pid: feat for pid, feat in pssm_dict.items() if pid in common_proteins}

    cos_adj_filtered, cos_proteins_filtered, cos2idx_filtered = filter_adjacency_matrix(cos_adj, cos_proteins,
                                                                                    common_proteins)
    lev_adj_filtered, lev_proteins_filtered, lev2idx_filtered = filter_adjacency_matrix(lev_adj, lev_proteins,
                                                                                    common_proteins)
    dom_adj_filtered, dom_proteins_filtered, dom2idx_filtered = filter_adjacency_matrix(dom_adj, dom_proteins,
                                                                                    common_proteins)

    protein2idx = cos2idx_filtered
    n_proteins = len(protein2idx)

    print("\n--- 3. Network Fusion ---")
    fused_adj = fuse_similarity_networks(cos_adj_filtered, lev_adj_filtered, dom_adj_filtered, args.lambda_param,
                                   args.cos_net_path, args.lev_net_path, args.dom_net_path)
    print("\n--- 4. Feature/Label Preparation ---")
    protein_feat = np.zeros((n_proteins, args.pssm_dim), dtype=np.float32)
    for pid, idx in protein2idx.items():
        protein_feat[idx] = filtered_pssm_dict[pid]
    protein_labels = load_protein_labels(args.label_path, protein2idx)
    print(f"Features: {protein_feat.shape}, Labels: {protein_labels.shape}")
    print("\n--- 5. Hypergraph Building ---")
    hyper_builder_knn = ProteinHypergraphBuilder(args.k)
    protein_hypergraph_knn = hyper_builder_knn.build_hypergraph_from_adj(fused_adj, protein2idx)
    print(f"KNN Hypergraph Nodes: {protein_hypergraph_knn.num_nodes()}, Edges: {protein_hypergraph_knn.num_edges()}")

    n_clusters_spectral = args.n_clusters_spectral
    hyper_builder_spectral = ProteinHypergraphBuilderSpectral(n_clusters=n_clusters_spectral)
    protein_hypergraph_spectral = hyper_builder_spectral.build_hypergraph_from_similarity(fused_adj, protein2idx)
    print(f"Spectral Hypergraph Nodes: {protein_hypergraph_spectral.num_nodes()}, Edges: {protein_hypergraph_spectral.num_edges()}")

    print("\n--- 6. Data Splitting and Tensor Conversion ---")
    indices = np.arange(n_proteins)

    # 将数据移动到 CPU (稍后在 train/test 函数中移动到 GPU)
    protein_feat_th = th.tensor(protein_feat, device='cpu', dtype=th.float32)
    protein_labels_th = th.tensor(protein_labels, device='cpu', dtype=th.float32)

    print(f"Total samples: {len(indices)}")

    print("\n--- 7. Model Initialization ---")
    criterion = nn.MSELoss()

    print("\n--- 8. Cross-Validation Training ---")
    fold_results = cross_validate(HypergraphProteinRegressionModel, protein_feat_th, protein_labels_th, indices, 
                                criterion, args, protein_hypergraph_knn, protein_hypergraph_spectral)

    print("\nFinal Cross-Validation Results Summary:")
    for result in fold_results:
        print(f"Fold {result['fold']}: Test MSE={result['test_mse']:.4f}, Test RMSE={result['test_rmse']:.4f}, Test MAE={result['test_mae']:.4f}, Test R²={result['test_r2']:.4f}")

if __name__ == "__main__":
    main()