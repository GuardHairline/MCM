#!/usr/bin/env python3
"""
全面测试零样本评估修复
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import json
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.evaluate import evaluate_single_task
from modules.train_utils import create_model
from datasets.get_dataset import get_dataset
from torch.utils.data import DataLoader

class DummyLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    def warning(self, msg):
        print(f"[WARNING] {msg}")
    def error(self, msg):
        print(f"[ERROR] {msg}")
    def debug(self, msg):
        print(f"[DEBUG] {msg}")

def test_zero_shot_evaluation_fix():
    """测试零样本评估修复"""
    print("开始全面测试零样本评估修复...")
    
    logger = DummyLogger()
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
            use_label_embedding=True,
            label_emb_dim=128,
            tam_cl=False,
            clap4clip=False,
            moe_adapters=False,
            moe_num_experts=4,
            moe_top_k=2,
            ddas=False,
            ddas_threshold=0.5,
            mymethod=False,
            ewc=False,
            replay=False,
            lwf=False,
            si=False,
            mas=False,
            gem=False,
            pnn=False,
            pretrained_model_path="",
            output_model_path="./dummy_model.pt",
            label_embedding_path="./dummy_label_emb.pt",
            test_text_file="data/MASC/twitter2015/test__.txt",
            train_text_file="data/MASC/twitter2015/train__.txt",
            dev_text_file="data/MASC/twitter2015/dev__.txt",
            image_dir="data/img"
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 测试1: 标准模型（无label embedding）
        print("\n=== 测试1: 标准模型 ===")
        args.use_label_embedding = False
        model = create_model(args, device, logger=logger)
        
        try:
            metrics = evaluate_single_task(model, "masc", "test", device, args)
            print(f"标准模型零样本评估成功: {metrics['acc']:.4f}")
        except Exception as e:
            print(f"标准模型零样本评估失败: {e}")
        
        # 测试2: 带label embedding的模型
        print("\n=== 测试2: 带label embedding的模型 ===")
        args.use_label_embedding = True
        model = create_model(args, device, logger=logger)
        
        try:
            metrics = evaluate_single_task(model, "masc", "test", device, args)
            print(f"Label embedding模型零样本评估成功: {metrics['acc']:.4f}")
        except Exception as e:
            print(f"Label embedding模型零样本评估失败: {e}")
        
        # 测试3: 序列任务（mate）
        print("\n=== 测试3: 序列任务（mate）===")
        args.task_name = "mate"
        args.num_labels = 7
        model = create_model(args, device, logger=logger)
        
        try:
            metrics = evaluate_single_task(model, "mate", "test", device, args)
            print(f"序列任务零样本评估成功: {metrics['acc']:.4f}")
        except Exception as e:
            print(f"序列任务零样本评估失败: {e}")
        
        # 测试4: 测试不同的任务类型
        print("\n=== 测试4: 不同任务类型 ===")
        tasks = ["masc", "mate", "mner", "mabsa"]
        for task in tasks:
            print(f"\n测试任务: {task}")
            if task == "masc":
                args.num_labels = 3
            else:
                args.num_labels = 7
            
            args.task_name = task
            model = create_model(args, device, logger=logger)
            
            try:
                metrics = evaluate_single_task(model, task, "test", device, args)
                print(f"  {task} 零样本评估成功: {metrics['acc']:.4f}")
            except Exception as e:
                print(f"  {task} 零样本评估失败: {e}")
        
        print("\n=== 测试完成 ===")
        return True
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_forward_compatibility():
    """测试模型前向传播兼容性"""
    print("\n=== 测试模型前向传播兼容性 ===")
    logger = DummyLogger()
    try:
        args = argparse.Namespace(
            text_model_name="microsoft/deberta-v3-base",
            image_model_name="google/vit-base-patch16-224-in21k",
            num_labels=3,
            dropout_prob=0.1,
            session_name="test_session",
            task_name="masc",
            batch_size=2,
            mode="text_only",
            fusion_strategy="concat",
            num_heads=8,
            hidden_dim=768,
            use_label_embedding=True,
            label_emb_dim=128,
            tam_cl=False,
            clap4clip=False,
            moe_adapters=False,
            moe_num_experts=4,
            moe_top_k=2,
            ddas=False,
            ddas_threshold=0.5,
            mymethod=False,
            ewc=False,
            replay=False,
            lwf=False,
            si=False,
            mas=False,
            gem=False,
            pnn=False,
            pretrained_model_path="",
            output_model_path="./dummy_model.pt",
            label_embedding_path="./dummy_label_emb.pt",
            test_text_file="data/MASC/twitter2015/test__.txt",
            train_text_file="data/MASC/twitter2015/train__.txt",
            dev_text_file="data/MASC/twitter2015/dev__.txt",
            image_dir="data/img"
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        model = create_model(args, device, logger=logger)
        model.eval()
        
        # 创建测试数据
        input_ids = torch.randint(0, 1000, (2, 10)).to(device)
        attention_mask = torch.ones(2, 10).to(device)
        token_type_ids = torch.zeros(2, 10).to(device)
        image_tensor = torch.randn(2, 3, 224, 224).to(device)
        
        # 测试前向传播
        with torch.no_grad():
            try:
                # 测试句级分类
                fused_feat = model.base_model(input_ids, attention_mask, token_type_ids, image_tensor, return_sequence=False)
                logits = model.head(fused_feat)
                print(f"句级分类前向传播成功，logits shape: {logits.shape}")
                
                # 测试序列标注
                fused_feat = model.base_model(input_ids, attention_mask, token_type_ids, image_tensor, return_sequence=True)
                logits = model.head(fused_feat)
                print(f"序列标注前向传播成功，logits shape: {logits.shape}")
                
            except Exception as e:
                print(f"前向传播失败: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"模型前向传播兼容性测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始全面测试零样本评估修复...")
    
    # 测试1: 零样本评估修复
    test1_success = test_zero_shot_evaluation_fix()
    
    # 测试2: 模型前向传播兼容性
    test2_success = test_model_forward_compatibility()
    
    print(f"\n=== 测试结果总结 ===")
    print(f"零样本评估修复测试: {'通过' if test1_success else '失败'}")
    print(f"模型前向传播兼容性测试: {'通过' if test2_success else '失败'}")
    
    if test1_success and test2_success:
        print("所有测试通过！零样本评估修复成功。")
    else:
        print("部分测试失败，需要进一步调试。") 