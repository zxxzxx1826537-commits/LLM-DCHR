# utils.py
import os
import pandas as pd
import numpy as np
from pyexpat import features
from scipy.sparse import coo_matrix
from tqdm import tqdm
import hashlib
import pickle


def load_protein_pssm(pssm_dir):
    """
    加载蛋白质PSSM特征（文档1-48：蛋白质特征数据处理逻辑）
    输入：PSSM特征文件夹路径（含10026个xxx.pssm文件）
    输出：dict{蛋白质ID: 400维PSSM特征向量（np.float32）}
    """
    pssm_dict = {}
    # 遍历文件夹下所有.pssm文件（文档1-48：PSSM为蛋白质核心特征）
    for filename in tqdm(os.listdir(pssm_dir), desc="Loading PSSM features (bbac159.pdf 1-48)"):
        # if not filename.endswith(".pssm"):
        #     continue  # 仅处理.pssm格式文件

        # 提取蛋白质ID（文件名去掉.pssm后缀）
        protein_id = filename.replace(".pssm", "")
        # 读取PSSM特征（文档1-48：每行1个400维向量，无表头，空格/逗号分隔）
        file_path = os.path.join(pssm_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if not line:
                raise ValueError(f"Empty PSSM file: {filename} (bbac159.pdf 1-48)")

            # 转换为400维浮点向量（文档1-48：固定PSSM特征维度为400）
            try:
                # 检查是逗号分隔还是空格分隔
                if ',' in line:
                    # 逗号分隔格式
                    feat = np.array([float(x) for x in line.split(',')], dtype=np.float32)
                else:
                    # 空格分隔格式
                    feat = np.array([float(x) for x in line.split()], dtype=np.float32)
            except ValueError:
                raise ValueError(f"Invalid PSSM format in {filename} (bbac159.pdf 1-48)")

            # 校验特征维度（文档1-48：确保特征一致性）
            if len(feat) != 1024:
                raise ValueError(f"PSSM dimension of {protein_id} is {len(feat)}, not 400 (bbac159.pdf 1-48)")

        pssm_dict[protein_id] = feat

    return pssm_dict


    #
    # pssm_dict = {}
    #
    # print(f"Loading PSSM features from {pssm_dir}")
    #
    # files = [f for f in os.listdir(pssm_dir) if f.endswith('.npy')]
    # for filename in tqdm(files, desc="Loading PSSM features"):
    #     filepath = os.path.join(pssm_dir, filename)
    #     protein_id = filename.split('.')[0]
    #
    #     try:
    #         # 使用NumPy加载.npy文件
    #         feature = np.load(filepath)
    #         feature=np.ravel( feature)
    #         pssm_dict[protein_id] = feature
    #     except Exception as e:
    #         print(f"Error loading {filename}: {e}")
    #
    # print(f"Loaded {len(pssm_dict)} PSSM features 并且平铺成向量")
    # return pssm_dict


# 全局缓存变量
_similarity_network_cache = {}
_fused_network_cache = {}

def load_similarity_network(net_path, chunksize=5000000):
    """
    加载蛋白质相似网络（文档1-38~1-41：相似网络构建逻辑）
    输入：相似网络CSV路径、Top-k邻居数（默认20，文档1-41策略）
    输出：(all_proteins, protein2idx, adj_matrix)
        - all_proteins: 排序后的蛋白质ID列表
        - protein2idx: 蛋白质ID→索引的映射（dict）
        - adj_matrix: 相似邻接矩阵（n×n，np.float32，保留所有邻居+自环）
    """
    # 检查缓存
    cache_key = net_path
    if cache_key in _similarity_network_cache:
        print(f"从内存缓存加载相似网络: {net_path}")
        return _similarity_network_cache[cache_key]

    print(f"加载相似网络: {net_path}")

    # 一次性读取所有数据以提高性能
    print("读取相似网络数据...")
    df = pd.read_csv(net_path)
    
    # 检查列名并进行适配
    if 'Protein_ID' in df.columns and 'Similar_Protein' in df.columns:
        # 标准格式
        protein_col1 = 'Protein_ID'
        protein_col2 = 'Similar_Protein'
        similarity_col = 'Similarity_Score' if 'Similarity_Score' in df.columns else 'Similarity'
    elif 'Protein1' in df.columns and 'Protein2' in df.columns:
        # Domain网络格式
        protein_col1 = 'Protein1'
        protein_col2 = 'Protein2'
        similarity_col = 'Similarity_Score' if 'Similarity_Score' in df.columns else 'Similarity'
    else:
        raise ValueError(f"Unknown column format in {net_path}")

    # 获取所有唯一蛋白质ID（文档1-38：确保网络覆盖完整）
    all_proteins = sorted(list(set(df[protein_col1].unique()).union(set(df[protein_col2].unique()))))
    protein2idx = {pid: idx for idx, pid in enumerate(all_proteins)}
    n_proteins = len(all_proteins)
    print(f"总共找到 {n_proteins} 个唯一蛋白质")

    # 向量化操作构建稀疏邻接矩阵（文档1-39：无向网络，双向添加边）
    print("构建邻接矩阵...")
    
    # 将蛋白质ID映射为索引
    protein1_indices = df[protein_col1].map(protein2idx).values
    protein2_indices = df[protein_col2].map(protein2idx).values
    similarities = df[similarity_col].values
    
    # 构建无向网络：双向添加边
    row = np.concatenate([protein1_indices, protein2_indices])
    col = np.concatenate([protein2_indices, protein1_indices])
    data = np.concatenate([similarities, similarities])
    
    print(f"总共处理了 {len(df)} 条边")

    # 转换为稠密矩阵并添加自环（文档1-79：自环相似度设为1.0，增强自身特征权重）
    adj_matrix = coo_matrix((data, (row, col)), shape=(n_proteins, n_proteins)).todense()
    np.fill_diagonal(adj_matrix, 1.0)  # 自环处理（文档1-79）

    # 改进对称性处理
    adj_matrix = (adj_matrix + adj_matrix.T) / 2.0

    result = (all_proteins, protein2idx, adj_matrix.astype(np.float32))

    # 缓存结果
    _similarity_network_cache[cache_key] = result

    return result


def filter_adjacency_matrix(adj_matrix, original_proteins, common_proteins):
    """
    过滤邻接矩阵，只保留共同蛋白质（向量化优化版本）
    """
    # 创建原始蛋白质到索引的映射
    original_pid2idx = {pid: idx for idx, pid in enumerate(original_proteins)}

    # 过滤出共同蛋白质并排序
    filtered_proteins = sorted([pid for pid in original_proteins if pid in common_proteins])
    protein2idx = {pid: idx for idx, pid in enumerate(filtered_proteins)}

    # 使用向量化操作提取子矩阵
    original_indices = [original_pid2idx[pid] for pid in filtered_proteins]
    filtered_adj = adj_matrix[np.ix_(original_indices, original_indices)]

    return filtered_adj, filtered_proteins, protein2idx



def fuse_similarity_networks(cos_adj, lev_adj, dom_adj, lambda_param=0.5,
                           cos_path=None, lev_path=None, dom_path=None):
    """
    融合三个蛋白质相似网络（文档1-38：多模态网络融合逻辑+用户指定公式）
    输入：
        - cos_adj: Cosine相似邻接矩阵（n×n）
        - lev_adj: Levenshtein相似邻接矩阵（n×n）
        - dom_adj: Domain相似邻接矩阵（n×n）
        - lambda_param: 融合权重（0-1，用户指定）
    输出：融合后的邻接矩阵（n×n，np.float32，归一化后）
    """
    # 校验输入矩阵维度一致性（文档1-38：确保多网络覆盖相同蛋白质）
    n_cos, _ = cos_adj.shape
    n_lev, _ = lev_adj.shape
    n_dom, _ = dom_adj.shape
    if not (n_cos == n_lev == n_dom):
        raise ValueError("All similarity networks must have the same dimension (bbac159.pdf 1-38)")

    # 检查内存缓存
    cache_key = (id(cos_adj), id(lev_adj), id(dom_adj), lambda_param)
    if cache_key in _fused_network_cache:
        print(f"从内存缓存加载融合网络，lambda_param={lambda_param}")
        return _fused_network_cache[cache_key]

    # 用户指定融合公式：lambda*(Lev + Cos)/2 + (1-lambda)*Domain（文档1-38多网络整合扩展）
    fused_adj = lambda_param * (lev_adj + cos_adj) / 2.0 + (1 - lambda_param) * dom_adj

    # 缓存到内存
    _fused_network_cache[cache_key] = fused_adj.astype(np.float32)

    print("Fused similarity network shape:", fused_adj.shape)
    print("Fused similarity network memory usage:", fused_adj)
    return fused_adj.astype(np.float32)


def load_protein_labels(label_path, protein2idx, fill_missing=0.0):
    """
    加载蛋白质回归标签（文档1-48：标签数据处理逻辑）
    输入：
        - label_path: 标签CSV路径（格式：Protein_ID,Label）
        - protein2idx: 蛋白质ID→索引的映射（dict）
        - fill_missing: 未匹配蛋白质的标签填充值（默认0.0）
    输出：标签数组（n×1，np.float32，与蛋白质索引对应）
    """
    # 读取标签CSV（文档1-48：标签与蛋白质ID一一对应）
    df = pd.read_csv(label_path)
    n_proteins = len(protein2idx)
    labels = np.full(n_proteins, fill_missing, dtype=np.float32)

    # 分配标签（文档1-48：确保标签与蛋白质索引对齐）
    for _, row in df.iterrows():
        pid = row["Protein_ID"]
        label = row["Label"]
        # 仅处理在相似网络中存在的蛋白质（文档1-48数据一致性）
        if pid in protein2idx:
            labels[protein2idx[pid]] = float(label)

    # # 找出非填充值的有效标签
    # valid_mask = labels != fill_missing
    # print(f"有效标签数量: {np.sum(valid_mask)}")
    # if np.any(valid_mask):
    #     valid_labels = labels[valid_mask]
    #     max_label = np.max(valid_labels)
    #     print(f"最大标签值: {max_label}")
    #
    #     # 使用最大值进行归一化
    #     if max_label != 0:  # 避免除零错误
    #         labels[valid_mask] = valid_labels / max_label

    return labels
