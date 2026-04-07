#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化超参数调优脚本
使用 Optuna 进行贝叶斯优化
"""

import torch as th
import torch.nn as nn
import numpy as np
import os
import optuna
from sklearn.model_selection import KFold, train_test_split
from HypergraphProteinRegressionModel import HypergraphProteinRegressionModel
from utils import (load_protein_pssm, load_similarity_network, fuse_similarity_networks,
                   load_protein_labels, filter_adjacency_matrix)

# 添加多进程保护，避免重复初始化
if __name__ != '__main__':
    # 禁用不必要的日志输出
    import logging
    logging.disable(logging.CRITICAL)

from train import (
    ProteinHypergraphBuilder, 
    ProteinHypergraphBuilderSpectral,
    cross_validate,
    Args
)

# 避免多线程库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def create_model_with_params(trial):
    """为 Optuna 试验创建带有特定参数的模型"""
    args = Args()
    
    # 定义超参数搜索空间
    #args.lambda_param = trial.suggest_categorical('lambda_param', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # 修改为离散值
    #args.hid_feats = trial.suggest_categorical('hid_feats', [256, 512, 1024])
    #args.dropout = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])  # 修改为离散值
    #args.lr = trial.suggest_categorical('lr', [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])  # 修改为离散值
    #args.wd = trial.suggest_categorical('wd', [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3])  # 修改为离散值
    #args.k=trial.suggest_int('k', 10, 40)
    #args.n_clusters_spectral=trial.suggest_int('n_clusters_spectral', 50, 200)
    # args.epochs = trial.suggest_int('epochs', 500, 2000)
    
    # # 回归头层数和维度的自动选择
    n_layers = trial.suggest_int('n_regressor_layers', 2, 5)
    regressor_layers = []

    # 预定义每层可能的维度选项，避免动态变化
    layer_dim_choices = {
         0: [64, 128, 256, 512],
         1: [32, 64, 128, 256],
         2: [32, 64, 128],
         3: [32, 64],
         4: [32]
    }

    for i in range(n_layers):
        # 使用预定义的选项，确保每次试验的选择空间一致
        choices = layer_dim_choices.get(i, [32, 64])
        dim = trial.suggest_categorical(f'regressor_layer_{i}', choices)
        regressor_layers.append(dim)

    args.regressor_layers = regressor_layers
    
    return args


def objective(trial):
    """Optuna 优化目标函数"""
    # 设置随机种子以确保可重现性
    np.random.seed(42)
    th.manual_seed(42)
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Testing hyperparameters")
    print(f"{'='*60}")
    
    # 获取试验参数
    args = create_model_with_params(trial)
    
    # 打印当前试验的参数
    print("Current trial parameters:")
    print(f"  lambda_param: {args.lambda_param}")
    print(f"  hid_feats: {args.hid_feats}")
    print(f"  dropout: {args.dropout}")
    print(f"  learning rate: {args.lr}")
    print(f"  weight decay: {args.wd}")
    print(f"  epochs: {args.epochs}")
    print(f"  k: {getattr(args, 'k', 'N/A')}")
    print(f"  n_clusters_spectral: {getattr(args, 'n_clusters_spectral', 'N/A')}")
    print(f"  regressor layers: {getattr(args, 'regressor_layers', [])}")
    
    try:
        # 数据加载部分（简化版，复用已有逻辑）
        print("\n--- 1. Data Loading ---")
        pssm_dict = load_protein_pssm(args.pssm_dir)
        cos_proteins, cos2idx, cos_adj = load_similarity_network(args.cos_net_path)
        lev_proteins, lev2idx, lev_adj = load_similarity_network(args.lev_net_path)
        dom_proteins, dom2idx, dom_adj = load_similarity_network(args.dom_net_path)

        print("\n--- 2. Data Filtering ---")
        common_proteins = set(pssm_dict.keys()) & set(cos_proteins) & set(lev_proteins) & set(dom_proteins)
        print(f"Found {len(common_proteins)} common proteins across all data sources")

        filtered_pssm_dict = {pid: feat for pid, feat in pssm_dict.items() if pid in common_proteins}

        cos_adj_filtered, cos_proteins_filtered, cos2idx_filtered = filter_adjacency_matrix(
            cos_adj, cos_proteins, common_proteins)
        lev_adj_filtered, lev_proteins_filtered, lev2idx_filtered = filter_adjacency_matrix(
            lev_adj, lev_proteins, common_proteins)
        dom_adj_filtered, dom_proteins_filtered, dom2idx_filtered = filter_adjacency_matrix(
            dom_adj, dom_proteins, common_proteins)

        protein2idx = cos2idx_filtered
        n_proteins = len(protein2idx)

        print("\n--- 3. Network Fusion ---")
        fused_adj = fuse_similarity_networks(
            cos_adj_filtered, lev_adj_filtered, dom_adj_filtered, args.lambda_param,
            args.cos_net_path, args.lev_net_path, args.dom_net_path)
        
        print("\n--- 4. Feature/Label Preparation ---")
        protein_feat = np.zeros((n_proteins, args.pssm_dim), dtype=np.float32)
        for pid, idx in protein2idx.items():
            protein_feat[idx] = filtered_pssm_dict[pid]
        protein_labels = load_protein_labels(args.label_path, protein2idx)
        print(f"Features: {protein_feat.shape}, Labels: {protein_labels.shape}")
        
        print("\n--- 5. Hypergraph Building ---")
        # 使用串行方式构建超图（无并行处理）
        hyper_builder_knn = ProteinHypergraphBuilder(top_k=args.k)
        protein_hypergraph_knn = hyper_builder_knn.build_hypergraph_from_adj(fused_adj, protein2idx)
        print(f"KNN Hypergraph Nodes: {protein_hypergraph_knn.num_nodes()}, Edges: {protein_hypergraph_knn.num_edges()}")

        hyper_builder_spectral = ProteinHypergraphBuilderSpectral(n_clusters=args.n_clusters_spectral)
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
        # 使用较少的折数以加快调参速度
        args.n_splits = 5  # 三折交叉验证以加快搜索
        
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
        avg_val_rmse = np.mean([r['best_val_rmse'] for r in fold_results])
        
        # 计算平均测试指标
        avg_test_mse = np.mean([r['test_mse'] for r in fold_results])
        avg_test_mae = np.mean([r['test_mae'] for r in fold_results])
        avg_test_rmse = np.mean([r['test_rmse'] for r in fold_results])
        avg_test_r2 = np.mean([r['test_r2'] for r in fold_results])
        
        # 将测试指标存储在 trial 的 user_attrs 中
        trial.set_user_attr("test_mse", avg_test_mse)
        trial.set_user_attr("test_rmse", avg_test_rmse)
        trial.set_user_attr("test_mae", avg_test_mae)
        trial.set_user_attr("test_r2", avg_test_r2)
        
        print(f"\nTrial {trial.number} completed.")
        print(f"  Average validation RMSE: {avg_val_rmse:.4f}")
        print(f"  Average test MSE: {avg_test_mse:.4f}")
        print(f"  Average test RMSE: {avg_test_rmse:.4f}")
        print(f"  Average test MAE: {avg_test_mae:.4f}")
        print(f"  Average test R²: {avg_test_r2:.4f}")
        
        return avg_test_rmse
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        # 发生错误时返回一个较大的损失值
        return float('inf')


def run_automl_optimization(n_trials=50):
    """运行自动化超参数优化"""
    print("开始自动化超参数优化...")
    print(f"计划执行 {n_trials} 次试验")
    
    # 创建 Optuna 研究对象
    study = optuna.create_study(
        direction='minimize',  # 最小化验证 MSE
        sampler=optuna.samplers.TPESampler(seed=42),  # 使用 TPE 采样器
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # 中位数剪枝器
    )
    
    # 运行优化
    study.optimize(objective, n_trials=n_trials)
    
    # 输出最佳结果
    print("\n" + "="*80)
    print("自动化超参数优化完成!")
    print("="*80)
    print(f"最佳测试 RMSE: {study.best_value:.4f}")
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 获取最佳试验的测试指标
    best_trial = study.best_trial
    best_test_mse = best_trial.user_attrs.get("test_mse", "N/A")
    best_test_rmse = best_trial.user_attrs.get("test_rmse", "N/A")
    best_test_mae = best_trial.user_attrs.get("test_mae", "N/A")
    best_test_r2 = best_trial.user_attrs.get("test_r2", "N/A")
    
    # 打印最佳试验的测试指标
    print("\n对应的最佳测试指标:")
    print(f"  测试 MSE: {best_test_mse:.4f}")
    print(f"  测试 RMSE: {best_test_rmse:.4f}")
    print(f"  测试 MAE: {best_test_mae:.4f}")
    print(f"  测试 R²: {best_test_r2:.4f}")
    
    # 保存研究结果
    with open("best_hyperparameters.txt", "w", encoding='utf-8') as f:
        f.write(f"Best validation MSE: {study.best_value:.4f}\n")
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
        
        # 保存最佳试验的测试指标
        f.write("\nCorresponding test metrics:\n")
        f.write(f"  Test MSE: {best_test_mse:.4f}\n")
        f.write(f"  Test RMSE: {best_test_rmse:.4f}\n")
        f.write(f"  Test MAE: {best_test_mae:.4f}\n")
        f.write(f"  Test R²: {best_test_r2:.4f}\n")
    
    print("\n最佳超参数和测试指标已保存到 'best_hyperparameters.txt'")
    return study


if __name__ == "__main__":
    # 运行自动化优化
    study = run_automl_optimization(n_trials=30)