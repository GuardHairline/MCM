#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版特征聚类可视化模块

新功能：
1. 使用实际标签名（NEG/NEU/POS等）而不是Class 0/1/2
2. 同时显示真实标签和预测标签
3. 标记预测错误的样本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 设置默认字体（无需中文支持）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def get_label_names(task_name: str) -> Dict[int, str]:
    """
    获取任务的实际标签名称
    
    Args:
        task_name: 任务名称
        
    Returns:
        label_names: {label_id: label_name}
    """
    from continual.label_config import get_label_manager
    
    manager = get_label_manager()
    config = manager.get_task_config(task_name)
    
    if config is None:
        logger.warning(f"Task {task_name} not found in label manager")
        return {}
    
    # 返回标签ID到标签名的映射
    label_names = {}
    for idx, name in enumerate(config.label_names):
        label_names[idx] = name
    
    return label_names


def extract_features_labels_and_predictions(
    model,
    task_name: str,
    split: str,
    device: torch.device,
    args,
    max_samples: int = 2000,
    extract_predictions: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    提取特征、真实标签和预测标签
    
    Args:
        model: 训练好的模型
        task_name: 任务名称
        split: 数据集划分
        device: 设备
        args: 参数
        max_samples: 最大样本数
        extract_predictions: 是否提取预测标签
        
    Returns:
        features: (N, hidden_dim) 融合后的特征
        true_labels: (N,) 真实标签（ground truth）
        pred_labels: (N,) 预测标签（如果extract_predictions=True）
    """
    from datasets.get_dataset import get_dataset
    from modules.train_utils import is_sequence_task
    
    logger.info(f"📊 提取特征和标签: task={task_name}, split={split}")
    
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
    
    dataset = get_dataset(task_name, split, args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    is_seq_task = is_sequence_task(task_name)
    
    all_features = []
    all_true_labels = []
    all_pred_labels = [] if extract_predictions else None
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
            true_labels = batch["labels"]
            
            # 1. 提取融合特征
            if is_seq_task:
                fused_feat = model.base_model(
                    input_ids, attention_mask, token_type_ids, image_tensor,
                    return_sequence=True
                )
                
                # 2. 获取预测（如果需要）
                if extract_predictions:
                    logits = model.head(fused_feat)  # (batch, seq_len, num_classes)
                    predictions = torch.argmax(logits, dim=-1)  # (batch, seq_len)
                    predictions = predictions.view(-1)
                
                # 展平
                batch_size, seq_len, hidden_dim = fused_feat.shape
                fused_feat = fused_feat.view(-1, hidden_dim)
                true_labels = true_labels.view(-1)
                
                # 过滤padding
                valid_mask = true_labels != -100
                fused_feat = fused_feat[valid_mask]
                true_labels = true_labels[valid_mask]
                if extract_predictions:
                    predictions = predictions[valid_mask]
            else:
                fused_feat = model.base_model(
                    input_ids, attention_mask, token_type_ids, image_tensor,
                    return_sequence=False
                )
                
                # 2. 获取预测（如果需要）
                if extract_predictions:
                    logits = model.head(fused_feat)  # (batch, num_classes)
                    predictions = torch.argmax(logits, dim=-1)
            
            # 保存
            all_features.append(fused_feat.cpu().numpy())
            all_true_labels.append(true_labels.cpu().numpy())
            if extract_predictions:
                all_pred_labels.append(predictions.cpu().numpy())
            
            sample_count += fused_feat.shape[0]
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  已处理 {batch_idx + 1}/{len(loader)} batches, {sample_count} samples")
    
    # 合并
    features = np.concatenate(all_features, axis=0)[:max_samples]
    true_labels = np.concatenate(all_true_labels, axis=0)[:max_samples]
    pred_labels = np.concatenate(all_pred_labels, axis=0)[:max_samples] if extract_predictions else None
    
    if extract_predictions:
        accuracy = np.mean(true_labels == pred_labels) * 100
        logger.info(f"✓ 特征提取完成: shape={features.shape}, accuracy={accuracy:.2f}%")
    else:
        logger.info(f"✓ 特征提取完成: shape={features.shape}")
    
    return features, true_labels, pred_labels


def plot_tsne_with_label_names(
    features: np.ndarray,
    labels: np.ndarray,
    task_name: str,
    save_path: str,
    label_names: Optional[Dict[int, str]] = None,
    title: str = None,
    perplexity: int = 30,
    n_iter: int = 1000
):
    """
    使用实际标签名绘制t-SNE图
    
    Args:
        features: (N, hidden_dim)
        labels: (N,) 标签ID
        task_name: 任务名称
        save_path: 保存路径
        label_names: {label_id: label_name} 标签名映射
        title: 图表标题
        perplexity: t-SNE参数
        n_iter: 迭代次数
    """
    logger.info(f"🎨 开始t-SNE降维并绘图: perplexity={perplexity}, n_iter={n_iter}")
    
    # 如果没有提供label_names，自动获取
    if label_names is None:
        label_names = get_label_names(task_name)
    
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
        
        # 获取标签名
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
        title = f't-SNE: {task_name.upper()} (Ground Truth Labels)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=10, ncol=2 if n_classes > 5 else 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ t-SNE图已保存: {save_path}")


def plot_tsne_comparison(
    features: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    task_name: str,
    save_path: str,
    label_names: Optional[Dict[int, str]] = None,
    perplexity: int = 30,
    n_iter: int = 1000
):
    """
    绘制对比图：真实标签 vs 预测标签
    
    Args:
        features: (N, hidden_dim)
        true_labels: (N,) 真实标签
        pred_labels: (N,) 预测标签
        task_name: 任务名称
        save_path: 保存路径
        label_names: 标签名映射
        perplexity: t-SNE参数
        n_iter: 迭代次数
    """
    logger.info(f"🎨 绘制真实标签 vs 预测标签对比图")
    
    # 如果没有提供label_names，自动获取
    if label_names is None:
        label_names = get_label_names(task_name)
    
    # t-SNE降维（只做一次）
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # 计算准确率
    accuracy = np.mean(true_labels == pred_labels) * 100
    correct_mask = true_labels == pred_labels
    
    # 获取唯一标签
    unique_labels = np.unique(true_labels)
    n_classes = len(unique_labels)
    colors = plt.cm.get_cmap('tab10' if n_classes <= 10 else 'tab20')(range(n_classes))
    
    # === 子图1: 真实标签 ===
    for idx, label in enumerate(unique_labels):
        mask = true_labels == label
        label_text = label_names.get(label, f'Class {label}') if label_names else f'Class {label}'
        
        ax1.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[idx]],
            label=label_text,
            alpha=0.6,
            s=30,
            edgecolors='k',
            linewidths=0.3
        )
    
    ax1.set_title(f'Ground Truth Labels\n(Expected Distribution)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax1.legend(loc='best', fontsize=9, ncol=2 if n_classes > 5 else 1)
    ax1.grid(True, alpha=0.3)
    
    # === 子图2: 预测标签（标记错误）===
    # 先绘制正确预测的点
    for idx, label in enumerate(unique_labels):
        mask = (pred_labels == label) & correct_mask
        label_text = label_names.get(label, f'Class {label}') if label_names else f'Class {label}'
        
        if np.any(mask):
            ax2.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[idx]],
                label=label_text,
                alpha=0.6,
                s=30,
                edgecolors='k',
                linewidths=0.3
            )
    
    # 再绘制错误预测的点（用X标记）
    if np.any(~correct_mask):
        ax2.scatter(
            features_2d[~correct_mask, 0],
            features_2d[~correct_mask, 1],
            c='red',
            label=f'Errors ({np.sum(~correct_mask)})',
            alpha=0.8,
            s=60,
            marker='x',
            linewidths=2
        )
    
    ax2.set_title(f'Predicted Labels (Accuracy: {accuracy:.2f}%)\n(Model Predictions)', 
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax2.legend(loc='best', fontsize=9, ncol=2 if n_classes > 5 else 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ 对比图已保存: {save_path}")
    logger.info(f"  准确率: {accuracy:.2f}%, 错误样本数: {np.sum(~correct_mask)}")


def visualize_task_enhanced(
    model,
    task_name: str,
    session_name: str,
    device: torch.device,
    args,
    save_dir: str,
    split: str = 'dev',
    max_samples: int = 2000,
    show_predictions: bool = True,
    config_name: Optional[str] = None  # 新增：配置文件名称前缀
):
    """
    增强版可视化：使用实际标签名，并对比真实vs预测
    
    Args:
        model: 训练好的模型
        task_name: 任务名称
        session_name: 会话名称
        device: 设备
        args: 参数
        save_dir: 保存目录
        split: 数据集划分
        max_samples: 最大样本数
        show_predictions: 是否生成预测对比图
        config_name: 配置文件名称（用于区分不同配置的可视化结果）
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建文件名前缀（避免不同配置互相覆盖）
    if config_name:
        file_prefix = f"{config_name}_{session_name}"
        logger.info(f"{'='*60}")
        logger.info(f"📊 增强版特征聚类可视化")
        logger.info(f"  配置: {config_name}")
        logger.info(f"  任务: {task_name}")
        logger.info(f"  会话: {session_name}")
        logger.info(f"  数据集: {split}")
        logger.info(f"  文件前缀: {file_prefix}")
        logger.info(f"{'='*60}\n")
    else:
        file_prefix = session_name
        logger.info(f"{'='*60}")
        logger.info(f"📊 增强版特征聚类可视化")
        logger.info(f"  任务: {task_name}")
        logger.info(f"  会话: {session_name}")
        logger.info(f"  数据集: {split}")
        logger.info(f"{'='*60}\n")
    
    # 1. 提取特征、真实标签和预测标签
    features, true_labels, pred_labels = extract_features_labels_and_predictions(
        model, task_name, split, device, args, max_samples, 
        extract_predictions=show_predictions
    )
    
    # 2. 获取标签名
    label_names = get_label_names(task_name)
    logger.info(f"✓ 标签名映射: {label_names}")
    
    # 3. 绘制真实标签图（使用实际标签名）
    tsne_path = save_dir / f'{file_prefix}_{split}_tsne_true.png'
    plot_tsne_with_label_names(
        features, true_labels, task_name, str(tsne_path),
        label_names=label_names,
        title=f't-SNE: {task_name.upper()} - Ground Truth ({split})'
    )
    
    # 4. 绘制对比图（真实 vs 预测）
    if show_predictions and pred_labels is not None:
        comparison_path = save_dir / f'{file_prefix}_{split}_tsne_comparison.png'
        plot_tsne_comparison(
            features, true_labels, pred_labels, task_name, str(comparison_path),
            label_names=label_names
        )
    
    # 5. 保存特征
    feature_save_path = save_dir / f'{file_prefix}_{split}_features_enhanced.npz'
    np.savez(feature_save_path, 
             features=features, 
             true_labels=true_labels, 
             pred_labels=pred_labels if pred_labels is not None else np.array([]),
             label_names=np.array(list(label_names.items()), dtype=object))
    logger.info(f"✓ 特征已保存: {feature_save_path}")
    
    logger.info(f"✓ 增强版可视化完成!\n")
    
    return features, true_labels, pred_labels


if __name__ == '__main__':
    print("增强版特征聚类可视化模块")
    print("新功能：")
    print("  1. 使用实际标签名（NEG/NEU/POS等）")
    print("  2. 对比真实标签 vs 预测标签")
    print("  3. 标记预测错误的样本")

