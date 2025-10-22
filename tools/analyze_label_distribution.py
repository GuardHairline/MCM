#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集标签分布分析工具
用于计算MASC、MNER等任务的真实类别分布和推荐权重
"""
import os
import json
import torch
from collections import Counter
import numpy as np
from datasets.get_dataset import get_dataset
import argparse

def analyze_dataset(task_name, dataset_name="twitter2015", split="train"):
    """分析数据集的标签分布"""
    print(f"\n{'='*80}")
    print(f"📊 分析任务: {task_name.upper()} | 数据集: {dataset_name} | 分割: {split}")
    print(f"{'='*80}\n")
    
    # 构造文件路径 (修正：任务目录在data下)
    # 情感任务（MASC, MATE, MABSA）都使用MASC数据（sentiment: -1, 0, 1）
    # 实体任务（MNER）使用MNER数据（NER标签）
    if task_name in ["masc", "mate", "mabsa"]:
        # 情感相关任务都使用MASC目录的数据
        # 注意：MASC/MATE/MABSA使用train__.txt（双下划线）
        base_path = f"data/MASC/{dataset_name}"
        text_file = f"{base_path}/{split}__.txt"
        image_dir = "data/img"
    elif task_name == "mner":
        # 命名实体识别任务使用MNER目录（双下划线）
        base_path = f"data/MNER/{dataset_name}"
        text_file = f"{base_path}/{split}__.txt"
        image_dir = "data/img"
    else:
        base_path = f"data/{dataset_name}"
        text_file = f"{base_path}/{split}.txt"
        image_dir = f"{base_path}/{split}"
    
    # 创建简单的args对象
    args = argparse.Namespace(
        task_name=task_name,
        dataset=dataset_name,
        train_text_file=text_file,
        dev_text_file=text_file.replace("train", "dev"),
        test_text_file=text_file.replace("train", "test"),
        image_dir=image_dir,  # 修复：添加image_dir属性
        text_model_name="microsoft/deberta-v3-base",
        image_model_name="openai/clip-vit-base-patch32",
        max_length=128,
        batch_size=32,
        deqa=0  # 不使用DEQA模式
    )
    
    try:
        # 加载数据集（修复：参数顺序是 task, split, args）
        # get_dataset返回的是单个dataset对象，不是tuple
        dataset = get_dataset(task_name, split, args)
        
        # 统计标签
        label_counter = Counter()
        total_tokens = 0  # 用于token级任务
        total_samples = len(dataset)
        
        print(f"数据集大小: {total_samples} 样本")
        
        for i, item in enumerate(dataset):
            labels = item['labels']
            
            if task_name in ["mner", "mate", "mabsa"]:
                # Token级任务：只统计非-100的标签
                # labels可能是tensor或list
                if isinstance(labels, torch.Tensor):
                    labels_list = labels.tolist()
                else:
                    labels_list = labels
                
                valid_labels = [l for l in labels_list if l != -100]
                label_counter.update(valid_labels)
                total_tokens += len(valid_labels)
                
                if i < 3:  # 显示前3个样本
                    print(f"样本 {i}: {len(valid_labels)} 个有效token")
            else:
                # 句级任务：直接统计
                if isinstance(labels, torch.Tensor):
                    labels = labels.item()
                label_counter[labels] += 1
        
        # 获取标签名称
        from continual.label_config import get_label_manager
        label_manager = get_label_manager()
        task_config = label_manager.get_task_config(task_name)
        if task_config is None:
            print(f"❌ 无法获取任务 {task_name} 的配置")
            return None
        label_names = task_config.label_names
        num_labels = task_config.num_labels
        
        print(f"标签总数: {num_labels}")
        print(f"标签名称: {label_names}")
        
        # 计算分布
        print(f"{'='*80}")
        print("📈 标签分布统计:")
        print(f"{'='*80}")
        
        counts = [label_counter.get(i, 0) for i in range(num_labels)]
        total = sum(counts)
        
        print(f"\n{'标签ID':<8} {'标签名':<15} {'数量':<12} {'占比':<10} {'推荐权重':<12}")
        print("-" * 80)
        
        weights_inverse = []
        weights_balanced = []
        weights_sqrt = []
        
        for i, (count, name) in enumerate(zip(counts, label_names)):
            if count == 0:
                print(f"{i:<8} {name:<15} {count:<12} {'0.00%':<10} {'N/A':<12}")
                weights_inverse.append(1.0)
                weights_balanced.append(1.0)
                weights_sqrt.append(1.0)
                continue
            
            pct = 100 * count / total
            
            # 计算多种权重策略
            # 1. 逆频率权重 (Inverse Frequency)
            inv_weight = total / (num_labels * count)
            weights_inverse.append(inv_weight)
            
            # 2. 平衡权重 (Balanced)
            balanced_weight = (total - count) / count
            weights_balanced.append(balanced_weight)
            
            # 3. 平方根逆频率 (sqrt inverse)
            sqrt_weight = np.sqrt(total / count)
            weights_sqrt.append(sqrt_weight)
            
            print(f"{i:<8} {name:<15} {count:<12} {pct:<9.2f}% {inv_weight:<12.2f}")
        
        # 归一化权重（使最小值为1.0）
        def normalize_weights(weights):
            min_w = min(w for w in weights if w > 0)
            return [w / min_w for w in weights]
        
        weights_inverse_norm = normalize_weights(weights_inverse)
        weights_balanced_norm = normalize_weights(weights_balanced)
        weights_sqrt_norm = normalize_weights(weights_sqrt)
        
        # 打印推荐权重
        print(f"{'='*80}")
        print("💡 推荐类别权重配置:")
        print(f"{'='*80}\n")
        
        print("方法1: 逆频率权重 (Inverse Frequency) - 适用于极度不平衡")
        print(f'"{task_name}": {[round(w, 1) for w in weights_inverse_norm]},')
        
        print("\n方法2: 平衡权重 (Balanced) - 适用于中等不平衡")
        print(f'"{task_name}": {[round(w, 1) for w in weights_balanced_norm]},')
        
        print("\n方法3: 平方根逆频率 (Sqrt Inverse) - 温和平衡，推荐使用")
        print(f'"{task_name}": {[round(w, 1) for w in weights_sqrt_norm]},')
        
        # 特殊推荐：对于MASC和MNER
        if task_name == "masc":
            # MASC通常是3分类：negative, neutral, positive
            # NEU通常占多数
            max_count = max(counts)
            custom_weights = [max_count / max(c, 1) for c in counts]
            custom_weights_norm = normalize_weights(custom_weights)
            print("\n方法4: 自定义MASC权重（推荐）")
            print(f'"{task_name}": {[round(w, 1) for w in custom_weights_norm]},')
        
        if task_name == "mner":
            # MNER的O标签通常占绝大多数
            # 建议对O使用较小权重，对实体类型使用较大权重
            custom_weights = weights_sqrt_norm.copy()
            if counts[0] > sum(counts[1:]):  # 如果O标签是第一个且占多数
                custom_weights[0] = 0.1  # O标签使用很小的权重
            print("\n方法5: 自定义MNER权重（推荐）")
            print(f'"{task_name}": {[round(w, 2) for w in custom_weights]},')
        
        print(f"{'='*80}\n")
        
        return {
            'label_names': label_names,
            'counts': counts,
            'total': total,
            'weights_inverse': weights_inverse_norm,
            'weights_balanced': weights_balanced_norm,
            'weights_sqrt': weights_sqrt_norm,
        }
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """分析所有任务的标签分布"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all", 
                       help="任务名称: masc, mner, mate, mabsa, 或 all")
    parser.add_argument("--dataset", type=str, default="twitter2015",
                       help="数据集名称")
    parser.add_argument("--split", type=str, default="train",
                       help="数据分割: train, dev, test")
    args = parser.parse_args()
    
    if args.task == "all":
        tasks = ["masc", "mner", "mate", "mabsa"]
    else:
        tasks = [args.task]
    
    results = {}
    for task in tasks:
        result = analyze_dataset(task, args.dataset, args.split)
        if result:
            results[task] = result
    
    # 生成配置文件建议
    print("="*80)
    print("📝 完整配置建议（复制到 continual/label_config.py）:")
    print("="*80 + "\n")
    print("weights = {")
    for task, result in results.items():
        if result:
            weights = result['weights_sqrt']
            print(f'    "{task}": {[round(w, 1) for w in weights]},')
    print("}")


if __name__ == "__main__":
    main()

