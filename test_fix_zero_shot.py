#!/usr/bin/env python3
"""
测试零样本评估修复
"""

import sys
import os
import argparse
import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.evaluate import evaluate_single_task
from modules.train_utils import create_model
from datasets.get_dataset import get_dataset
from torch.utils.data import DataLoader

def test_zero_shot_evaluation():
    """测试零样本评估修复"""
    print("开始测试零样本评估修复...")
    
    try:
        # 创建测试参数
        args = argparse.Namespace(
            text_model_name="microsoft/deberta-v3-base",
            image_model_name="google/vit-base-patch16-224-in21k",
            num_labels=3,
            dropout_prob=0.1,
            session_name="test_session",
            task_name="masc",
            batch_size=4,
            mode="text_only",
            fusion_strategy="concat",
            num_heads=8,
            hidden_dim=768,
            image_dir="data/img",
            train_text_file="data/MASC/mix/train.txt",
            test_text_file="data/MASC/mix/test.txt",
            dev_text_file="data/MASC/mix/dev.txt",
            use_label_embedding=False,
            tam_cl=False,
            clap4clip=False,
            moe_adapters=False,
            ddas=False,
            num_workers=0
        )
        
        print("1. 创建测试参数成功")
        
        # 创建模型
        device = "cpu"
        model = create_model(args, device, label_embedding_manager=None, logger=None)
        print("2. 模型创建成功")
        
        # 创建测试数据集
        test_dataset = get_dataset("masc", "test", args)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        print("3. 测试数据集创建成功")
        
        # 测试零样本评估
        print("4. 开始零样本评估测试...")
        metrics = evaluate_single_task(model, "masc", "test", device, args)
        print(f"   评估成功，准确率: {metrics['acc']:.2f}%")
        
        print("\n🎉 零样本评估修复测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tam_cl_zero_shot():
    """测试TAM-CL模型的零样本评估"""
    print("\n开始测试TAM-CL零样本评估...")
    
    try:
        # 创建TAM-CL测试参数
        args = argparse.Namespace(
            text_model_name="microsoft/deberta-v3-base",
            image_model_name="google/vit-base-patch16-224-in21k",
            num_labels=3,
            dropout_prob=0.1,
            session_name="test_session",
            task_name="masc",
            batch_size=4,
            mode="text_only",
            fusion_strategy="concat",
            num_heads=8,
            hidden_dim=768,
            image_dir="data/img",
            train_text_file="data/MASC/mix/train.txt",
            test_text_file="data/MASC/mix/test.txt",
            dev_text_file="data/MASC/mix/dev.txt",
            use_label_embedding=False,
            tam_cl=True,
            clap4clip=False,
            moe_adapters=False,
            ddas=False,
            num_workers=0
        )
        
        print("1. 创建TAM-CL测试参数成功")
        
        # 创建TAM-CL模型
        device = "cpu"
        model = create_model(args, device, label_embedding_manager=None, logger=None)
        print("2. TAM-CL模型创建成功")
        
        # 创建测试数据集
        test_dataset = get_dataset("masc", "test", args)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        print("3. 测试数据集创建成功")
        
        # 测试零样本评估
        print("4. 开始TAM-CL零样本评估测试...")
        metrics = evaluate_single_task(model, "masc", "test", device, args)
        print(f"   TAM-CL评估成功，准确率: {metrics['acc']:.2f}%")
        
        print("\n🎉 TAM-CL零样本评估修复测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ TAM-CL测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 零样本评估修复测试 ===")
    
    # 测试标准模型
    success1 = test_zero_shot_evaluation()
    
    # 测试TAM-CL模型
    success2 = test_tam_cl_zero_shot()
    
    if success1 and success2:
        print("\n✅ 所有测试通过！零样本评估修复成功。")
    else:
        print("\n❌ 部分测试失败，需要进一步检查。") 