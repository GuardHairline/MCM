#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征聚类可视化模块
用于观察持续学习过程中的表示变化和类别分布

功能：
1. 提取模型中间层特征（融合后的特征）
2. 使用t-SNE/UMAP降维到2D
3. 可视化聚类情况（按类别和任务着色）
4. 观察持续学习中的遗忘情况
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

# 设置默认字体（无需中文支持）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def extract_features_and_labels(
    model,
    task_name: str,
    split: str,
    device: torch.device,
    args,
    max_samples: int = 2000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    提取特征和标签
    
    Args:
        model: 训练好的模型
        task_name: 任务名称 (e.g., "mate", "mabsa")
        split: 数据集划分 ("train", "dev", "test")
        device: 设备
        args: 参数
        max_samples: 最大样本数（避免内存溢出）
        
    Returns:
        features: (N, hidden_dim) 提取的特征
        labels: (N,) 标签
        task_ids: (N,) 任务ID（用于跨任务可视化）
    """
    from datasets.get_dataset import get_dataset
    from modules.train_utils import is_sequence_task
    
    logger.info(f"📊 提取特征: task={task_name}, split={split}, max_samples={max_samples}")
    
    # 设置模型为评估模式
    model.eval()
    
    # 确保使用正确的任务头
    if hasattr(model, 'set_active_head') and hasattr(args, 'session_name'):
        try:
            model.set_active_head(args.session_name, strict=False)
        except Exception as e:
            logger.warning(f"Failed to set active head: {e}")
    
    # 确保base_model的mode正确设置
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'mode'):
        current_mode = getattr(args, 'mode', 'multimodal')
        model.base_model.mode = current_mode
    
    # 加载数据集
    dataset = get_dataset(task_name, split, args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    is_seq_task = is_sequence_task(task_name)
    
    all_features = []
    all_labels = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if sample_count >= max_samples:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            image_tensor = batch["image_tensor"].to(device)
            labels = batch["labels"]
            
            # 提取融合后的特征（在任务头之前）
            if is_seq_task:
                # Token级别任务：返回序列特征
                fused_feat = model.base_model(
                    input_ids, attention_mask, token_type_ids, image_tensor,
                    return_sequence=True
                )  # (batch_size, seq_len, hidden_dim)
                
                # 展平序列，只保留有效token（非padding）
                batch_size, seq_len, hidden_dim = fused_feat.shape
                fused_feat = fused_feat.view(-1, hidden_dim)  # (batch_size * seq_len, hidden_dim)
                labels = labels.view(-1)  # (batch_size * seq_len,)
                
                # 过滤掉padding（label=-100）
                valid_mask = labels != -100
                fused_feat = fused_feat[valid_mask]
                labels = labels[valid_mask]
                
            else:
                # 句子级别任务：返回CLS特征
                fused_feat = model.base_model(
                    input_ids, attention_mask, token_type_ids, image_tensor,
                    return_sequence=False
                )  # (batch_size, hidden_dim)
            
            # 转为numpy并保存
            all_features.append(fused_feat.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            sample_count += fused_feat.shape[0]
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  已处理 {batch_idx + 1}/{len(loader)} batches, {sample_count} samples")
    
    # 合并所有批次
    features = np.concatenate(all_features, axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples]
    
    logger.info(f"✓ 特征提取完成: shape={features.shape}, unique_labels={len(np.unique(labels))}")
    
    return features, labels


def get_label_names_for_task(task_name: str) -> Dict[int, str]:
    """
    获取任务的实际标签名称
    
    Args:
        task_name: 任务名称
        
    Returns:
        label_names: {label_id: label_name}
    """
    try:
        from continual.label_config import get_label_manager
        
        manager = get_label_manager()
        config = manager.get_task_config(task_name)
        
        if config is None:
            logger.warning(f"Task {task_name} not found in label manager, using generic names")
            return {}
        
        # 返回标签ID到标签名的映射
        label_names = {}
        for idx, name in enumerate(config.label_names):
            label_names[idx] = name
        
        return label_names
    except Exception as e:
        logger.warning(f"Failed to get label names: {e}, using generic names")
        return {}


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    task_name: str,
    save_path: str,
    title: str = None,
    perplexity: int = 30,
    n_iter: int = 1000
):
    """
    使用t-SNE降维并绘制2D散点图（使用实际标签名）
    
    Args:
        features: (N, hidden_dim)
        labels: (N,)
        task_name: 任务名称
        save_path: 保存路径
        title: 图表标题
        perplexity: t-SNE perplexity参数
        n_iter: t-SNE迭代次数
    """
    logger.info(f"🎨 开始t-SNE降维: perplexity={perplexity}, n_iter={n_iter}")
    
    # 获取实际标签名
    label_names = get_label_names_for_task(task_name)
    
    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # 绘图
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # 获取唯一标签
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # 生成颜色映射
    colors = plt.cm.get_cmap('tab10' if n_classes <= 10 else 'tab20')(range(n_classes))
    
    # 为每个类别绘制散点
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        
        # 使用实际标签名或默认名称
        if label_names and label in label_names:
            label_text = label_names[label]
        else:
            label_text = f'Class {label}'
        
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[idx]],
            label=label_text,  # 使用实际标签名
            alpha=0.6,
            s=30,
            edgecolors='k',
            linewidths=0.3
        )
    
    # 设置标题和图例
    if title is None:
        title = f't-SNE: {task_name.upper()} (Ground Truth)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=10, ncol=2 if n_classes > 5 else 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ t-SNE图已保存: {save_path}")


def plot_umap(
    features: np.ndarray,
    labels: np.ndarray,
    task_name: str,
    save_path: str,
    title: str = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1
):
    """
    使用UMAP降维并绘制2D散点图（使用实际标签名）
    
    Args:
        features: (N, hidden_dim)
        labels: (N,)
        task_name: 任务名称
        save_path: 保存路径
        title: 图表标题
        n_neighbors: UMAP n_neighbors参数
        min_dist: UMAP min_dist参数
    """
    try:
        import umap
    except ImportError:
        logger.warning("⚠️  UMAP未安装，跳过UMAP可视化。请运行: pip install umap-learn")
        return
    
    logger.info(f"🎨 开始UMAP降维: n_neighbors={n_neighbors}, min_dist={min_dist}")
    
    # 获取实际标签名
    label_names = get_label_names_for_task(task_name)
    
    # UMAP降维
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    features_2d = reducer.fit_transform(features)
    
    # 绘图
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # 获取唯一标签
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # 生成颜色映射
    colors = plt.cm.get_cmap('tab10' if n_classes <= 10 else 'tab20')(range(n_classes))
    
    # 为每个类别绘制散点
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        
        # 使用实际标签名或默认名称
        if label_names and label in label_names:
            label_text = label_names[label]
        else:
            label_text = f'Class {label}'
        
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[idx]],
            label=label_text,  # 使用实际标签名
            alpha=0.6,
            s=30,
            edgecolors='k',
            linewidths=0.3
        )
    
    # 设置标题和图例
    if title is None:
        title = f'UMAP: {task_name.upper()} (Ground Truth)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=10, ncol=2 if n_classes > 5 else 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ UMAP图已保存: {save_path}")


def plot_continual_learning_evolution(
    task_features: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_dir: str,
    method: str = 'tsne'
):
    """
    绘制持续学习过程中所有任务的特征演进图
    
    Args:
        task_features: {task_name: (features, labels)} 所有任务的特征
        save_dir: 保存目录
        method: 降维方法 ('tsne' 或 'umap')
    """
    logger.info(f"🎨 绘制持续学习演进图: {len(task_features)} 个任务, method={method}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 合并所有任务的特征
    all_features = []
    all_labels = []
    all_task_ids = []
    
    task_names = list(task_features.keys())
    for task_idx, task_name in enumerate(task_names):
        features, labels = task_features[task_name]
        all_features.append(features)
        all_labels.append(labels)
        all_task_ids.append(np.full(len(labels), task_idx))
    
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_task_ids = np.concatenate(all_task_ids, axis=0)
    
    logger.info(f"  合并后特征: {all_features.shape}")
    
    # 降维
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        features_2d = reducer.fit_transform(all_features)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
            features_2d = reducer.fit_transform(all_features)
        except ImportError:
            logger.warning("UMAP未安装，回退到t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
            features_2d = reducer.fit_transform(all_features)
            method = 'tsne'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 创建两个子图：(1) 按任务着色  (2) 按类别着色
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # === 子图1: 按任务着色 ===
    task_colors = plt.cm.get_cmap('Set1')(range(len(task_names)))
    for task_idx, task_name in enumerate(task_names):
        mask = all_task_ids == task_idx
        ax1.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[task_colors[task_idx]],
            label=f'{task_name.upper()}',
            alpha=0.6,
            s=15,
            edgecolors='k',
            linewidths=0.2
        )
    
    ax1.set_title(f'{method.upper()}: Colored by Task', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax1.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # === 子图2: 按类别着色 ===
    unique_labels = np.unique(all_labels)
    n_classes = len(unique_labels)
    label_colors = plt.cm.get_cmap('tab20' if n_classes > 10 else 'tab10')(range(n_classes))
    
    for idx, label in enumerate(unique_labels):
        mask = all_labels == label
        ax2.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[label_colors[idx]],
            label=f'Class {label}',
            alpha=0.6,
            s=15,
            edgecolors='k',
            linewidths=0.2
        )
    
    ax2.set_title(f'{method.upper()}: Colored by Label', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax2.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    if n_classes <= 15:
        ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / f'continual_learning_evolution_{method}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ 演进图已保存: {save_path}")


def visualize_task_after_training(
    model,
    task_name: str,
    session_name: str,
    device: torch.device,
    args,
    save_dir: str,
    split: str = 'dev',
    max_samples: int = 2000,
    use_both_methods: bool = True,
    config_name: Optional[str] = None  # 新增：配置文件名称前缀
):
    """
    在任务训练完成后进行可视化
    
    Args:
        model: 训练好的模型
        task_name: 任务名称
        session_name: 会话名称
        device: 设备
        args: 参数
        save_dir: 保存目录
        split: 数据集划分 (推荐使用'dev')
        max_samples: 最大样本数
        use_both_methods: 是否同时使用t-SNE和UMAP
        config_name: 配置文件名称（用于区分不同配置的可视化结果）
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建文件名前缀（避免不同配置互相覆盖）
    if config_name:
        file_prefix = f"{config_name}_{session_name}"
        logger.info(f"{'='*60}")
        logger.info(f"📊 开始特征聚类可视化")
        logger.info(f"  配置: {config_name}")
        logger.info(f"  任务: {task_name}")
        logger.info(f"  会话: {session_name}")
        logger.info(f"  数据集: {split}")
        logger.info(f"  保存目录: {save_dir}")
        logger.info(f"  文件前缀: {file_prefix}")
        logger.info(f"{'='*60}\n")
    else:
        file_prefix = session_name
        logger.info(f"{'='*60}")
        logger.info(f"📊 开始特征聚类可视化")
        logger.info(f"  任务: {task_name}")
        logger.info(f"  会话: {session_name}")
        logger.info(f"  数据集: {split}")
        logger.info(f"  保存目录: {save_dir}")
        logger.info(f"{'='*60}\n")
    
    # 1. 提取特征
    features, labels = extract_features_and_labels(
        model, task_name, split, device, args, max_samples
    )
    
    # 2. t-SNE可视化
    tsne_path = save_dir / f'{file_prefix}_{split}_tsne.png'
    plot_tsne(
        features, labels, task_name, str(tsne_path),
        title=f't-SNE: {task_name.upper()} ({split})'
    )
    
    # 3. UMAP可视化（可选）
    if use_both_methods:
        umap_path = save_dir / f'{file_prefix}_{split}_umap.png'
        plot_umap(
            features, labels, task_name, str(umap_path),
            title=f'UMAP: {task_name.upper()} ({split})'
        )
    
    # 4. 保存特征到文件（用于后续跨任务分析）
    feature_save_path = save_dir / f'{file_prefix}_{split}_features.npz'
    np.savez(feature_save_path, features=features, labels=labels)
    logger.info(f"✓ 特征已保存: {feature_save_path}")
    
    logger.info(f"✓ 可视化完成!\n")
    
    return features, labels


def visualize_all_tasks_evolution(
    save_dir: str,
    split: str = 'dev',
    method: str = 'tsne'
):
    """
    加载所有已保存的任务特征，绘制演进图
    
    Args:
        save_dir: 特征保存目录
        split: 数据集划分
        method: 降维方法
    """
    save_dir = Path(save_dir)
    
    # 查找所有特征文件
    feature_files = list(save_dir.glob(f'*_{split}_features.npz'))
    
    if len(feature_files) == 0:
        logger.warning(f"⚠️  未找到特征文件: {save_dir}/*_{split}_features.npz")
        return
    
    logger.info(f"📊 找到 {len(feature_files)} 个任务的特征文件")
    
    # 加载所有特征
    task_features = {}
    for feature_file in sorted(feature_files):
        # 从文件名提取任务名
        task_name = feature_file.stem.replace(f'_{split}_features', '')
        
        # 加载特征
        data = np.load(feature_file)
        features = data['features']
        labels = data['labels']
        
        task_features[task_name] = (features, labels)
        logger.info(f"  ✓ 加载: {task_name} - {features.shape[0]} samples")
    
    # 绘制演进图
    plot_continual_learning_evolution(task_features, str(save_dir), method=method)


if __name__ == '__main__':
    """
    示例用法
    """
    print("特征聚类可视化模块")
    print("请在训练脚本中调用相关函数")

