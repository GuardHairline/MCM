#!/usr/bin/env python3
"""
多模态持续学习0样本转移分析脚本

分析模型在不同任务间的知识转移效果，特别关注：
1. 0样本准确率 (Zero-Shot Accuracy)
2. 前向转移 (Forward Transfer)
3. 任务相似性转移
4. 多模态特征共享效果
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

from continual.metrics import ContinualMetrics, compute_multimodal_transfer_metrics, analyze_task_similarity_transfer


def load_train_info(train_info_path: str) -> Dict:
    """加载训练信息"""
    with open(train_info_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_zero_shot_transfer(train_info: Dict) -> Dict:
    """分析0样本转移效果"""
    cm = ContinualMetrics()
    cm.acc_matrix = train_info.get("acc_matrix", [])
    
    if len(cm.acc_matrix) < 2:
        print("需要至少2个任务才能分析转移效果")
        return {}
    
    task_names = [session.get('task_name', 'unknown') for session in train_info.get("sessions", [])]
    task_count = len(task_names)
    
    # 计算转移指标
    transfer_metrics = compute_multimodal_transfer_metrics(cm, task_count, task_names)
    similarity_analysis = analyze_task_similarity_transfer(cm, task_names)
    
    # 合并结果
    analysis_result = {
        "transfer_metrics": transfer_metrics,
        "similarity_analysis": similarity_analysis,
        "task_names": task_names,
        "acc_matrix": cm.acc_matrix
    }
    
    return analysis_result


def plot_transfer_matrix(acc_matrix: List[List[float]], task_names: List[str], save_path: Optional[str] = None):
    """绘制转移矩阵热力图"""
    if not acc_matrix or not task_names:
        print("数据不足，无法绘制转移矩阵")
        return
    
    # 转换为numpy数组
    matrix = np.array(acc_matrix)
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, 
                annot=True, 
                fmt='.3f',
                xticklabels=task_names,
                yticklabels=task_names,
                cmap='YlOrRd',
                cbar_kws={'label': 'Accuracy'})
    
    plt.title('Task Transfer Matrix', fontsize=16)
    plt.xlabel('Target Task', fontsize=12)
    plt.ylabel('Source Task', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"转移矩阵图已保存到: {save_path}")
    
    plt.show()


def plot_metrics_comparison(transfer_metrics: Dict, save_path: Optional[str] = None):
    """绘制指标对比图"""
    if not transfer_metrics:
        print("没有转移指标数据")
        return
    
    # 提取基础指标
    basic_metrics = ['AA', 'AIA', 'FM', 'BWT', 'FWT', 'ZS_ACC', 'CT']
    multimodal_metrics = ['text_task_transfer', 'ner_task_transfer', 'cross_type_transfer']
    
    # 准备数据
    basic_values = []
    basic_labels = []
    for metric in basic_metrics:
        if metric in transfer_metrics and transfer_metrics[metric] is not None:
            basic_values.append(transfer_metrics[metric])
            basic_labels.append(metric)
    
    multimodal_values = []
    multimodal_labels = []
    for metric in multimodal_metrics:
        if metric in transfer_metrics and transfer_metrics[metric] is not None:
            multimodal_values.append(transfer_metrics[metric])
            multimodal_labels.append(metric)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 基础指标
    if basic_values:
        bars1 = ax1.bar(basic_labels, basic_values, color='skyblue', alpha=0.7)
        ax1.set_title('Basic Continual Learning Metrics', fontsize=14)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars1, basic_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # 多模态转移指标
    if multimodal_values:
        bars2 = ax2.bar(multimodal_labels, multimodal_values, color='lightcoral', alpha=0.7)
        ax2.set_title('Multimodal Transfer Metrics', fontsize=14)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars2, multimodal_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"指标对比图已保存到: {save_path}")
    
    plt.show()


def print_analysis_report(analysis_result: Dict):
    """打印分析报告"""
    print("=" * 60)
    print("多模态持续学习0样本转移分析报告")
    print("=" * 60)
    
    task_names = analysis_result.get("task_names", [])
    print(f"任务序列: {' -> '.join(task_names)}")
    print(f"总任务数: {len(task_names)}")
    print()
    
    # 转移指标
    transfer_metrics = analysis_result.get("transfer_metrics", {})
    print("转移指标:")
    print("-" * 30)
    for metric, value in transfer_metrics.items():
        if value is not None:
            print(f"{metric:15s}: {value:.4f}")
        else:
            print(f"{metric:15s}: N/A")
    print()
    
    # 相似性分析
    similarity_analysis = analysis_result.get("similarity_analysis", {})
    if similarity_analysis:
        print("任务相似性分析:")
        print("-" * 30)
        for metric, value in similarity_analysis.items():
            print(f"{metric:25s}: {value:.4f}")
        print()
    
    # 0样本转移分析
    if "ZS_ACC" in transfer_metrics and transfer_metrics["ZS_ACC"] is not None:
        zs_acc = transfer_metrics["ZS_ACC"]
        print("0样本转移分析:")
        print("-" * 30)
        print(f"0样本准确率: {zs_acc:.4f}")
        if zs_acc > 0.5:
            print("✓ 模型具有良好的0样本泛化能力")
        elif zs_acc > 0.3:
            print("○ 模型具有一定的0样本泛化能力")
        else:
            print("✗ 模型的0样本泛化能力较弱")
        print()


def main():
    parser = argparse.ArgumentParser(description="分析多模态持续学习的0样本转移效果")
    parser.add_argument("--train_info", type=str, required=True,
                       help="训练信息JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="./analysis_results",
                       help="分析结果输出目录")
    parser.add_argument("--plot", action="store_true", default=True,
                       help="是否生成可视化图表")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 加载训练信息
    print(f"加载训练信息: {args.train_info}")
    train_info = load_train_info(args.train_info)
    
    # 分析转移效果
    print("分析0样本转移效果...")
    analysis_result = analyze_zero_shot_transfer(train_info)
    
    if not analysis_result:
        print("分析失败，请检查数据")
        return
    
    # 打印报告
    print_analysis_report(analysis_result)
    
    # 保存分析结果
    result_path = output_dir / "transfer_analysis.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    print(f"分析结果已保存到: {result_path}")
    
    # 生成可视化图表
    if args.plot:
        print("生成可视化图表...")
        
        # 转移矩阵图
        matrix_plot_path = output_dir / "transfer_matrix.png"
        plot_transfer_matrix(
            analysis_result.get("acc_matrix", []),
            analysis_result.get("task_names", []),
            str(matrix_plot_path)
        )
        
        # 指标对比图
        metrics_plot_path = output_dir / "metrics_comparison.png"
        plot_metrics_comparison(
            analysis_result.get("transfer_metrics", {}),
            str(metrics_plot_path)
        )
    
    print("分析完成！")


if __name__ == "__main__":
    main() 