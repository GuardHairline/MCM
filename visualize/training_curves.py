# visualize/training_curves.py
"""
训练过程可视化模块 - 绘制训练曲线图

功能：
- 绘制训练/验证 Loss 曲线
- 绘制 Span F1 / Accuracy 曲线
- 双Y轴图表
- 标记最佳性能点
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path

# 设置中文字体
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: str,
    task_name: str = "Training",
    figsize: tuple = (12, 6),
    dpi: int = 150,
    show_grid: bool = True
):
    """
    绘制训练过程曲线图（双Y轴）
    
    Args:
        metrics_history: 指标历史数据，格式：
            {
                'epochs': [1, 2, 3, ...],
                'train_loss': [loss1, loss2, ...],
                'dev_loss': [loss1, loss2, ...],
                'span_f1': [f1_1, f1_2, ...] 或 'dev_f1'
            }
        save_path: 保存路径（如 'output/training_curves.png'）
        task_name: 任务名称（用于标题）
        figsize: 图像大小
        dpi: 分辨率
        show_grid: 是否显示网格
    
    Returns:
        保存的文件路径
    """
    # 确保输出目录存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 提取数据
    epochs = metrics_history.get('epochs', list(range(1, len(metrics_history.get('train_loss', [])) + 1)))
    train_loss = metrics_history.get('train_loss', [])
    dev_loss = metrics_history.get('dev_loss', [])
    
    # F1 可能叫 'span_f1', 'dev_f1', 'f1', 'acc'
    f1_scores = (metrics_history.get('span_f1') or 
                 metrics_history.get('dev_f1') or 
                 metrics_history.get('f1') or 
                 metrics_history.get('acc') or 
                 metrics_history.get('dev_acc') or 
                 [])
    
    # 检查数据完整性
    if not train_loss or not dev_loss:
        print(f"⚠️ 警告：缺少loss数据，无法绘制曲线")
        return None
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 左Y轴：Loss
    color_train = 'tab:blue'
    color_dev = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold', color='black')
    
    line1 = ax1.plot(epochs, train_loss, color=color_train, linewidth=2, 
                     marker='o', markersize=4, label='Train Loss', alpha=0.8)
    line2 = ax1.plot(epochs, dev_loss, color=color_dev, linewidth=2, 
                     marker='s', markersize=4, label='Dev Loss', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='black')
    
    # 找到最小loss点
    if dev_loss:
        min_loss_idx = np.argmin(dev_loss)
        min_loss_epoch = epochs[min_loss_idx]
        min_loss_value = dev_loss[min_loss_idx]
        ax1.plot(min_loss_epoch, min_loss_value, 'r*', markersize=15, 
                label=f'Min Dev Loss @ Epoch {min_loss_epoch}', zorder=5)
    
    # 右Y轴：F1 / Accuracy
    if f1_scores:
        ax2 = ax1.twinx()
        color_f1 = 'tab:green'
        ax2.set_ylabel('F1 Score / Accuracy (%)', fontsize=12, fontweight='bold', color=color_f1)
        
        line3 = ax2.plot(epochs, f1_scores, color=color_f1, linewidth=2.5, 
                        linestyle='--', marker='D', markersize=5, 
                        label='Span F1', alpha=0.9)
        ax2.tick_params(axis='y', labelcolor=color_f1)
        
        # 设置Y轴范围（0-100%）
        ax2.set_ylim([0, max(100, max(f1_scores) * 1.1)])
        
        # 找到最高F1点
        max_f1_idx = np.argmax(f1_scores)
        max_f1_epoch = epochs[max_f1_idx]
        max_f1_value = f1_scores[max_f1_idx]
        ax2.plot(max_f1_epoch, max_f1_value, 'g*', markersize=18, 
                label=f'Best F1 @ Epoch {max_f1_epoch}: {max_f1_value:.2f}%', zorder=5)
        
        # 添加F1文本标注
        ax2.annotate(f'{max_f1_value:.2f}%', 
                    xy=(max_f1_epoch, max_f1_value),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green'))
    else:
        ax2 = None
    
    # 标题
    plt.title(f'{task_name} - Training Progress', fontsize=14, fontweight='bold', pad=20)
    
    # 网格
    if show_grid:
        ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 合并图例
    lines = line1 + line2
    if ax2:
        lines += line3
    labels = [l.get_label() for l in lines]
    
    # 添加最佳点到图例
    if dev_loss:
        labels.append(f'Min Dev Loss @ Epoch {min_loss_epoch}: {min_loss_value:.4f}')
    if f1_scores:
        labels.append(f'Best F1 @ Epoch {max_f1_epoch}: {max_f1_value:.2f}%')
    
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.02, 0.98), 
              framealpha=0.9, fontsize=9)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()
    
    return save_path


def plot_multi_task_curves(
    all_metrics: Dict[str, Dict[str, List[float]]],
    save_dir: str,
    figsize: tuple = (15, 10),
    dpi: int = 150
):
    """
    绘制多任务训练曲线（子图）
    
    Args:
        all_metrics: {
            'task1': {'epochs': [...], 'train_loss': [...], 'dev_f1': [...]},
            'task2': {...},
            ...
        }
        save_dir: 保存目录
        figsize: 图像大小
        dpi: 分辨率
    
    Returns:
        保存的文件路径列表
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_files = []
    
    # 为每个任务绘制独立的图表
    for task_name, metrics in all_metrics.items():
        save_path = os.path.join(save_dir, f'{task_name}_training_curves.png')
        result = plot_training_curves(
            metrics_history=metrics,
            save_path=save_path,
            task_name=task_name.upper(),
            figsize=(10, 6),
            dpi=dpi
        )
        if result:
            saved_files.append(result)
    
    # 绘制综合对比图（可选）
    if len(all_metrics) > 1:
        plot_tasks_comparison(all_metrics, save_dir, figsize, dpi)
    
    return saved_files


def plot_tasks_comparison(
    all_metrics: Dict[str, Dict[str, List[float]]],
    save_dir: str,
    figsize: tuple = (15, 5),
    dpi: int = 150
):
    """
    绘制多任务F1对比图
    
    Args:
        all_metrics: 所有任务的指标
        save_dir: 保存目录
        figsize: 图像大小
        dpi: 分辨率
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # 左图：所有任务的F1曲线
    ax_f1 = axes[0]
    ax_f1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_f1.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax_f1.set_title('All Tasks F1 Comparison', fontsize=13, fontweight='bold')
    ax_f1.grid(True, alpha=0.3, linestyle='--')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_metrics)))
    
    for idx, (task_name, metrics) in enumerate(all_metrics.items()):
        epochs = metrics.get('epochs', list(range(1, len(metrics.get('dev_f1', [])) + 1)))
        f1_scores = (metrics.get('span_f1') or metrics.get('dev_f1') or 
                    metrics.get('f1') or metrics.get('acc') or [])
        if f1_scores:
            ax_f1.plot(epochs, f1_scores, color=colors[idx], linewidth=2, 
                      marker='o', markersize=4, label=task_name.upper(), alpha=0.8)
    
    ax_f1.legend(loc='lower right', framealpha=0.9)
    
    # 右图：最终F1对比柱状图
    ax_bar = axes[1]
    task_names = []
    final_f1s = []
    
    for task_name, metrics in all_metrics.items():
        f1_scores = (metrics.get('span_f1') or metrics.get('dev_f1') or 
                    metrics.get('f1') or metrics.get('acc') or [])
        if f1_scores:
            task_names.append(task_name.upper())
            final_f1s.append(max(f1_scores))
    
    if task_names:
        bars = ax_bar.bar(range(len(task_names)), final_f1s, color=colors[:len(task_names)], alpha=0.7)
        ax_bar.set_xlabel('Task', fontsize=12, fontweight='bold')
        ax_bar.set_ylabel('Best F1 Score (%)', fontsize=12, fontweight='bold')
        ax_bar.set_title('Best F1 Comparison', fontsize=13, fontweight='bold')
        ax_bar.set_xticks(range(len(task_names)))
        ax_bar.set_xticklabels(task_names, rotation=45, ha='right')
        ax_bar.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 添加数值标注
        for i, (bar, f1) in enumerate(zip(bars, final_f1s)):
            ax_bar.text(i, f1 + 1, f'{f1:.2f}%', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'tasks_comparison.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ 任务对比图已保存: {save_path}")
    plt.close()
    
    return save_path


def load_metrics_from_json(json_path: str) -> Dict[str, List[float]]:
    """
    从JSON文件加载训练指标
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        metrics_history字典
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 适配不同的JSON结构
    if 'metrics_history' in data:
        return data['metrics_history']
    else:
        return data


# ============================================================================
# 测试代码
# ============================================================================

def test_training_curves():
    """测试训练曲线绘制"""
    print("="*80)
    print("测试训练曲线绘制")
    print("="*80)
    
    # 模拟训练数据
    print("\n1. 生成模拟数据...")
    epochs = list(range(1, 21))
    train_loss = [2.5 - 0.1 * e + 0.05 * np.sin(e) for e in epochs]
    dev_loss = [2.6 - 0.08 * e + 0.08 * np.sin(e * 1.5) for e in epochs]
    span_f1 = [30 + 2.5 * e - 0.05 * e**2 + 2 * np.sin(e * 0.5) for e in epochs]
    
    metrics_history = {
        'epochs': epochs,
        'train_loss': train_loss,
        'dev_loss': dev_loss,
        'span_f1': span_f1
    }
    
    # 绘制单个任务曲线
    print("\n2. 绘制单个任务曲线...")
    save_path = 'tests/test_training_curve.png'
    plot_training_curves(
        metrics_history=metrics_history,
        save_path=save_path,
        task_name='MNER'
    )
    
    # 绘制多任务曲线
    print("\n3. 绘制多任务对比...")
    all_metrics = {
        'mate': metrics_history,
        'mner': {
            'epochs': epochs,
            'train_loss': [2.3 - 0.11 * e for e in epochs],
            'dev_loss': [2.4 - 0.09 * e for e in epochs],
            'span_f1': [25 + 3 * e - 0.06 * e**2 for e in epochs]
        },
        'mabsa': {
            'epochs': epochs,
            'train_loss': [2.7 - 0.09 * e for e in epochs],
            'dev_loss': [2.8 - 0.07 * e for e in epochs],
            'span_f1': [20 + 3.5 * e - 0.07 * e**2 for e in epochs]
        }
    }
    
    plot_multi_task_curves(
        all_metrics=all_metrics,
        save_dir='tests/multi_task_curves'
    )
    
    print("\n" + "="*80)
    print("✅ 所有测试通过！")
    print("="*80)


if __name__ == "__main__":
    test_training_curves()

