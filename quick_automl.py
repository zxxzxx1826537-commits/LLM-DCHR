#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速自动化超参数调优脚本
使用简化配置进行快速调参测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch as th
import torch.nn as nn
import numpy as np
import optuna
from sklearn.model_selection import KFold
from HypergraphProteinRegressionModel import HypergraphProteinRegressionModel
from utils import (load_protein_pssm, load_similarity_network, fuse_similarity_networks,
                   load_protein_labels, filter_adjacency_matrix)
from train import (
    ProteinHypergraphBuilder, 
    ProteinHypergraphBuilderSpectral,
    cross_validate,
    Args
)

# 避免多线程库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def quick_objective(trial):
    """快速优化目标函数"""
    # 设置随机种子以确保可重现性
    np.random.seed(42)
    th.manual_seed(42)
    
    print(f"\n{'='*60}")
    print(f"快速试验 {trial.number}: 测试超参数")
    print(f"{'='*60}")
    
    # 获取试验参数
    args = Args()
    
    # 快速试验使用较小的数据集和较少的训练轮次
    args.lambda_param = trial.suggest_float('lambda_param', 0.0, 1.0)
    args.hid_feats = trial.suggest_categorical('hid_feats', [256, 512])
    args.dropout = trial.suggest_float('dropout', 0.0, 0.3)
    args.lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    args.epochs = trial.suggest_int('epochs', 100, 500)
    
    # 回归头层数和维度的自动选择
    n_layers = trial.suggest_int('n_regressor_layers', 1, 3)
    regressor_layers = []
    base_dims = [args.hid_feats//2, args.hid_feats//4, 64]
    for i in range(n_layers):
        dim = trial.suggest_categorical(f'regressor_layer_{i}', 
                                       [base_dims[min(i, len(base_dims)-1)], 
                                        base_dims[min(i, len(base_dims)-1)]//2,
                                        32 if i == n_layers-1 else 64])
        regressor_layers.append(dim)
    
    args.regressor_layers = regressor_layers
    
    # 减少交叉验证折数以加快速度
    args.n_splits = 2
    
    print("当前试验参数:")
    print(f"  lambda_param: {args.lambda_param}")
    print(f"  hid_feats: {args.hid_feats}")
    print(f"  dropout: {args.dropout}")
    print(f"  learning rate: {args.lr}")
    print(f"  epochs: {args.epochs}")
    print(f"  regressor layers: {getattr(args, 'regressor_layers', [])}")
    
    try:
        # 为了快速测试，我们可以考虑只使用一部分数据
        print("\n--- 1. 数据加载 ---")
        pssm_dict = load_protein_pssm(args.pssm_dir)
        cos_proteins, cos2idx, cos_adj = load_similarity_network(args.cos_net_path)
        lev_proteins, lev2idx, lev_adj = load_similarity_network(args.lev_net_path)
        dom_proteins, dom2idx, dom_adj = load_similarity_network(args.dom_net_path)

        print("\n--- 2. 数据过滤 ---")
        common_proteins = set(pssm_dict.keys()) & set(cos_proteins) & set(lev_proteins) & set(dom_proteins)
        print(f"找到 {len(common_proteins)} 个共有蛋白质")

        # 为了快速测试，只使用前100个蛋白质
        common_proteins_limited = list(common_proteins)[:100] if len(common_proteins) > 100 else list(common_proteins)
        filtered_pssm_dict = {pid: feat for pid, feat in pssm_dict.items() if pid in common_proteins_limited}

        cos_adj_filtered, cos_proteins_filtered, cos2idx_filtered = filter_adjacency_matrix(
            cos_adj, cos_proteins, common_proteins_limited)
        lev_adj_filtered, lev_proteins_filtered, lev2idx_filtered = filter_adjacency_matrix(
            lev_adj, lev_proteins, common_proteins_limited)
        dom_adj_filtered, dom_proteins_filtered, dom2idx_filtered = filter_adjacency_matrix(
            dom_adj, dom_proteins, common_proteins_limited)

        protein2idx = cos2idx_filtered
        n_proteins = len(protein2idx)

        print("\n--- 3. 网络融合 ---")
        fused_adj = fuse_similarity_networks(
            cos_adj_filtered, lev_adj_filtered, dom_adj_filtered, args.lambda_param,
            args.cos_net_path, args.lev_net_path, args.dom_net_path)
        
        print("\n--- 4. 特征/标签准备 ---")
        protein_feat = np.zeros((n_proteins, args.pssm_dim), dtype=np.float32)
        for pid, idx in protein2idx.items():
            protein_feat[idx] = filtered_pssm_dict[pid]
        protein_labels = load_protein_labels(args.label_path, protein2idx)
        print(f"特征: {protein_feat.shape}, 标签: {protein_labels.shape}")
        
        print("\n--- 5. 超图构建 ---")
        hyper_builder_knn = ProteinHypergraphBuilder(top_k=10)  # 减少邻居数
        protein_hypergraph_knn = hyper_builder_knn.build_hypergraph_from_adj(fused_adj, protein2idx)
        print(f"KNN超图节点: {protein_hypergraph_knn.num_nodes()}, 边: {protein_hypergraph_knn.num_edges()}")

        n_clusters_spectral = min(int(n_proteins * 0.1) or 10, 20)  # 减少簇数
        hyper_builder_spectral = ProteinHypergraphBuilderSpectral(n_clusters=n_clusters_spectral)
        protein_hypergraph_spectral = hyper_builder_spectral.build_hypergraph_from_similarity(fused_adj, protein2idx)
        print(f"谱聚类超图节点: {protein_hypergraph_spectral.num_nodes()}, 边: {protein_hypergraph_spectral.num_edges()}")

        print("\n--- 6. 数据分割和张量转换 ---")
        indices = np.arange(n_proteins)

        # 将数据移动到 CPU (稍后在 train/test 函数中移动到 GPU)
        protein_feat_th = th.tensor(protein_feat, device='cpu', dtype=th.float32)
        protein_labels_th = th.tensor(protein_labels, device='cpu', dtype=th.float32)

        print(f"总样本数: {len(indices)}")

        print("\n--- 7. 模型初始化 ---")
        criterion = nn.MSELoss()

        print("\n--- 8. 交叉验证训练 ---")
        
        # 执行交叉验证
        fold_results = cross_validate(
            HypergraphProteinRegressionModel, 
            protein_feat_th, 
            protein_labels_th, 
            indices, 
            criterion, 
            args, 
            protein_hypergraph_knn, 
            protein_hypergraph_spectral
        )

        # 计算平均验证 MSE 作为目标值
        avg_val_mse = np.mean([r['best_val_mse'] for r in fold_results])
        
        print(f"\n试验 {trial.number} 完成. 平均验证 MSE: {avg_val_mse:.4f}")
        return avg_val_mse
        
    except Exception as e:
        print(f"试验 {trial.number} 失败，错误: {str(e)}")
        # 发生错误时返回一个较大的损失值
        return float('inf')


def run_quick_automl(n_trials=10):
    """运行快速自动化超参数优化"""
    print("开始快速自动化超参数优化...")
    print(f"计划执行 {n_trials} 次试验")
    
    # 创建 Optuna 研究对象
    study = optuna.create_study(
        direction='minimize',  # 最小化验证 MSE
        sampler=optuna.samplers.TPESampler(seed=42),  # 使用 TPE 采样器
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)  # 中位数剪枝器
    )
    
    # 运行优化
    study.optimize(quick_objective, n_trials=n_trials)
    
    # 输出最佳结果
    print("\n" + "="*80)
    print("快速自动化超参数优化完成!")
    print("="*80)
    print(f"最佳验证 MSE: {study.best_value:.4f}")
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 保存研究结果
    with open("quick_best_hyperparameters.txt", "w", encoding='utf-8') as f:
        f.write(f"最佳验证 MSE: {study.best_value:.4f}\n")
        f.write("最佳超参数:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    
    print("\n最佳超参数已保存到 'quick_best_hyperparameters.txt'")
    return study


if __name__ == "__main__":
    # 运行快速优化
    study = run_quick_automl(n_trials=10)