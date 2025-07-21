#!/usr/bin/env python3
"""
测试CLAP4CLIP模型
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continual.clap4clip import CLAP4CLIP, TaskAdapter, ProbabilisticFinetuning
from continual.clap4clip.clap_utils import (
    create_clap4clip_model,
    compute_clap4clip_loss,
    get_clip_processor,
    preprocess_clap4clip_data
)

def test_clap4clip_basic():
    """测试CLAP4CLIP基本功能"""
    print("开始测试CLAP4CLIP基本功能...")
    
    try:
        # 创建模型
        print("1. 创建CLAP4CLIP模型...")
        model = CLAP4CLIP(
            text_model_name="openai/clip-vit-base-patch32",
            image_model_name="openai/clip-vit-base-patch32",
            num_labels=3,
            dropout_prob=0.1,
            adapter_size=64,
            finetune_lambda=0.1,
            temperature=0.07
        )
        print("   ✓ CLAP4CLIP模型创建成功")
        
        # 测试任务管理
        print("2. 测试任务管理...")
        model.add_task("task1", 3)
        model.add_task("task2", 5)
        model.set_current_task("task1")
        print(f"   ✓ 任务管理正常，当前任务: {model.current_task}")
        print(f"   ✓ 任务头数量: {len(model.task_heads)}")
        
        # 创建测试数据
        print("3. 创建测试数据...")
        batch_size = 2
        seq_len = 77
        hidden_size = model.projection_dim
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        image_tensor = torch.randn(batch_size, 3, 224, 224)
        
        print("   ✓ 测试数据创建成功")
        
        # 测试前向传播
        print("4. 测试前向传播...")
        model.eval()
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_tensor=image_tensor,
                task_name="task1"
            )
        print(f"   ✓ 前向传播成功，输出形状: {logits.shape}")
        
        # 测试特征提取
        print("5. 测试特征提取...")
        text_features = model.get_text_features(input_ids, attention_mask)
        image_features = model.get_image_features(image_tensor)
        print(f"   ✓ 文本特征形状: {text_features.shape}")
        print(f"   ✓ 图像特征形状: {image_features.shape}")
        
        # 测试对比学习损失
        print("6. 测试对比学习损失...")
        contrastive_loss = model.compute_contrastive_loss(text_features, image_features)
        print(f"   ✓ 对比学习损失: {contrastive_loss.item():.4f}")
        
        print("\n🎉 CLAP4CLIP基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_probabilistic_finetuning():
    """测试概率微调模块"""
    print("\n开始测试概率微调模块...")
    
    try:
        # 创建概率微调模块
        print("1. 创建概率微调模块...")
        dummy_model = type('DummyModel', (), {'hidden_size': 512})()
        pf_module = ProbabilisticFinetuning(
            model=dummy_model,
            num_tasks=3,
            finetune_lambda=0.1,
            temperature=0.07,
            use_uncertainty=True
        )
        print("   ✓ 概率微调模块创建成功")
        
        # 创建测试特征
        print("2. 创建测试特征...")
        batch_size = 4
        hidden_size = 512
        
        text_features = torch.randn(batch_size, hidden_size)
        image_features = torch.randn(batch_size, hidden_size)
        
        print("   ✓ 测试特征创建成功")
        
        # 测试概率计算
        print("3. 测试概率计算...")
        task_probs = pf_module.compute_task_probabilities(text_features, image_features)
        print(f"   ✓ 任务概率形状: {task_probs.shape}")
        print(f"   ✓ 概率和: {task_probs.sum(dim=1)}")
        
        # 测试不确定性估计
        print("4. 测试不确定性估计...")
        uncertainty = pf_module.estimate_uncertainty(text_features + image_features)
        print(f"   ✓ 不确定性形状: {uncertainty.shape}")
        print(f"   ✓ 不确定性范围: [{uncertainty.min().item():.4f}, {uncertainty.max().item():.4f}]")
        
        # 测试概率微调更新
        print("5. 测试概率微调更新...")
        text_finetuned, image_finetuned, task_probs = pf_module.probabilistic_update(
            text_features, image_features
        )
        print(f"   ✓ 微调后文本特征形状: {text_finetuned.shape}")
        print(f"   ✓ 微调后图像特征形状: {image_finetuned.shape}")
        
        print("\n🎉 概率微调模块测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_clap4clip_integration():
    """测试CLAP4CLIP与训练流程的集成"""
    print("\n开始测试CLAP4CLIP集成...")
    
    try:
        # 创建参数
        print("1. 创建测试参数...")
        args = argparse.Namespace(
            text_model_name="openai/clip-vit-base-patch32",
            image_model_name="openai/clip-vit-base-patch32",
            num_labels=3,
            dropout_prob=0.1,
            adapter_size=64,
            finetune_lambda=0.1,
            temperature=0.07,
            session_name="test_session",
            clap4clip=True,
            pretrained_model_path=None,
            use_label_embedding=False,
            tam_cl=False,
            moe_adapters=False
        )
        
        print("   ✓ 测试参数创建成功")
        
        # 测试模型创建
        print("2. 测试模型创建...")
        from modules.train_utils import create_model
        
        device = "cpu"
        model = create_model(args, device, label_embedding_manager=None, logger=None)
        print("   ✓ 模型创建成功")
        
        # 测试前向传播
        print("3. 测试集成前向传播...")
        batch_size = 2
        seq_len = 77
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        image_tensor = torch.randn(batch_size, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_tensor=image_tensor
            )
        print(f"   ✓ 集成前向传播成功，输出形状: {logits.shape}")
        
        print("\n🎉 CLAP4CLIP集成测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("CLAP4CLIP模型测试")
    print("=" * 60)
    
    # 测试基本功能
    success1 = test_clap4clip_basic()
    
    # 测试概率微调
    success2 = test_probabilistic_finetuning()
    
    # 测试集成
    success3 = test_clap4clip_integration()
    
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"基本功能测试: {'✓ 通过' if success1 else '✗ 失败'}")
    print(f"概率微调测试: {'✓ 通过' if success2 else '✗ 失败'}")
    print(f"集成测试: {'✓ 通过' if success3 else '✗ 失败'}")
    
    if success1 and success2 and success3:
        print("\n🎉 所有测试通过！CLAP4CLIP模型可以正常使用。")
    else:
        print("\n❌ 部分测试失败，需要进一步调试。")
    
    print("=" * 60) 