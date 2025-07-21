#!/usr/bin/env python3
"""
测试正方形准确率矩阵功能
"""

import json
import numpy as np
from continual.metrics import ContinualMetrics, compute_metrics_example

def test_square_matrix():
    """测试正方形准确率矩阵功能"""
    
    # 创建一个模拟的准确率矩阵
    # 假设有3个任务：masc, mate, mner
    # 矩阵格式：[训练任务][测试任务]
    # 对角线及以下：训练后的准确率
    # 对角线上方：0样本转移准确率
    
    # 示例数据：
    # 第1行：训练masc后，在masc上的准确率，在mate上的0样本准确率，在mner上的0样本准确率
    # 第2行：训练mate后，在masc上的准确率，在mate上的准确率，在mner上的0样本准确率  
    # 第3行：训练mner后，在masc上的准确率，在mate上的准确率，在mner上的准确率
    
    acc_matrix = [
        [85.0, 45.0, 30.0],  # 训练masc后：masc=85%, mate=45%(0样本), mner=30%(0样本)
        [75.0, 90.0, 35.0],  # 训练mate后：masc=75%, mate=90%, mner=35%(0样本)
        [70.0, 85.0, 88.0]   # 训练mner后：masc=70%, mate=85%, mner=88%
    ]
    
    print("=== 正方形准确率矩阵测试 ===")
    print("矩阵结构：")
    print("行：训练任务")
    print("列：测试任务")
    print("对角线及以下：训练后的准确率")
    print("对角线上方：0样本转移准确率")
    print()
    
    # 打印矩阵
    print("准确率矩阵：")
    tasks = ["masc", "mate", "mner"]
    print("      ", end="")
    for task in tasks:
        print(f"{task:>8}", end="")
    print()
    
    for i, row in enumerate(acc_matrix):
        print(f"{tasks[i]:>6}", end="")
        for j, acc in enumerate(row):
            if j <= i:
                print(f"{acc:>8.1f}", end="")  # 训练后的准确率
            else:
                print(f"{acc:>8.1f}*", end="")  # 0样本转移准确率
        print()
    print("* 表示0样本转移准确率")
    print()
    
    # 创建ContinualMetrics实例
    cm = ContinualMetrics()
    cm.acc_matrix = acc_matrix
    
    # 计算各项指标
    print("=== 持续学习指标 ===")
    for k in range(1, 4):
        metrics = compute_metrics_example(cm, k)
        print(f"任务{k} ({tasks[k-1]}) 的指标：")
        for metric_name, value in metrics.items():
            if value is not None:
                print(f"  {metric_name}: {value:.2f}")
            else:
                print(f"  {metric_name}: None")
        print()
    
    # 测试0样本转移指标
    print("=== 0样本转移分析 ===")
    for k in range(1, 4):
        zs_transfer = cm.get_zero_shot_transfer_metrics(k)
        if zs_transfer is not None:
            print(f"任务{k} ({tasks[k-1]}) 的0样本转移准确率: {zs_transfer:.2f}%")
        else:
            print(f"任务{k} ({tasks[k-1]}) 的0样本转移准确率: None")
    print()
    
    # 保存和加载测试
    print("=== 保存和加载测试 ===")
    test_file = "test_acc_matrix.json"
    cm.save_to_json(test_file)
    print(f"准确率矩阵已保存到: {test_file}")
    
    # 加载测试
    cm2 = ContinualMetrics()
    cm2.load_from_json(test_file)
    print("准确率矩阵加载成功")
    print(f"加载的矩阵: {cm2.acc_matrix}")
    
    # 清理测试文件
    import os
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"测试文件已清理: {test_file}")

def test_real_scenario():
    """测试真实场景：模拟训练过程"""
    
    print("\n=== 真实场景测试 ===")
    
    cm = ContinualMetrics()
    
    # 模拟训练第1个任务（masc）
    print("训练第1个任务 (masc)...")
    performance_list = [85.0]  # masc的准确率
    zero_shot_metrics = {
        "mate": {"acc": 45.0},  # 对mate的0样本准确率
        "mner": {"acc": 30.0}   # 对mner的0样本准确率
    }
    cm.update_acc_matrix(0, performance_list, zero_shot_metrics)
    print(f"第1行: {cm.acc_matrix[0]}")
    
    # 模拟训练第2个任务（mate）
    print("训练第2个任务 (mate)...")
    performance_list = [75.0, 90.0]  # masc和mate的准确率
    zero_shot_metrics = {
        "mner": {"acc": 35.0}  # 对mner的0样本准确率
    }
    cm.update_acc_matrix(1, performance_list, zero_shot_metrics)
    print(f"第2行: {cm.acc_matrix[1]}")
    
    # 模拟训练第3个任务（mner）
    print("训练第3个任务 (mner)...")
    performance_list = [70.0, 85.0, 88.0]  # masc、mate、mner的准确率
    zero_shot_metrics = {}  # 没有后续任务了
    cm.update_acc_matrix(2, performance_list, zero_shot_metrics)
    print(f"第3行: {cm.acc_matrix[2]}")
    
    print("\n最终准确率矩阵:")
    for i, row in enumerate(cm.acc_matrix):
        print(f"任务{i+1}: {row}")
    
    # 计算最终指标
    print("\n最终持续学习指标:")
    metrics = compute_metrics_example(cm, 3)
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"  {metric_name}: {value:.2f}")
        else:
            print(f"  {metric_name}: None")

if __name__ == "__main__":
    test_square_matrix()
    test_real_scenario() 