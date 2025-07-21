#!/usr/bin/env python3
"""
测试MOE模型与label embedding的修复
"""

import sys
import os
import argparse
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.train_utils import create_model
from continual.label_embedding_manager import LabelEmbeddingManager

def test_moe_label_embedding_compat():
    """测试MOE模型与label embedding的兼容性"""
    print("开始测试MOE模型与label embedding的兼容性...")
    
    # 构造参数
    args = argparse.Namespace(
        text_model_name="microsoft/deberta-v3-base",
        image_model_name="google/vit-base-patch16-224-in21k",
        fusion_strategy="concat",
        num_heads=8,
        mode="text_only",
        task_name="masc",
        session_name="masc_1",
        num_labels=3,
        dropout_prob=0.1,
        moe_adapters=True,
        moe_num_experts=4,
        moe_top_k=2,
        use_label_embedding=True,
        label_emb_dim=128,
        use_similarity_reg=True,
        similarity_weight=0.1,
        pretrained_model_path=None,
        ddas=False,
        ddas_threshold=0.5,
        tam_cl=False,
        clap4clip=False,
    )

    device = "cpu"
    
    try:
        # 创建label embedding manager
        print("1. 创建LabelEmbeddingManager...")
        label_embedding_manager = LabelEmbeddingManager(
            emb_dim=args.label_emb_dim,
            use_similarity_regularization=args.use_similarity_reg,
            similarity_weight=args.similarity_weight
        )
        # 创建或加载标签嵌入
        label_embedding_manager.create_or_load_embedding(device=device)
        print("   ✓ LabelEmbeddingManager创建成功")
        
        # 创建模型
        print("2. 创建MOE模型...")
        model = create_model(args, device, label_embedding_manager, logger=None)
        print("   ✓ MOE模型创建成功")
        
        # 检查任务头管理功能
        print("3. 检查任务头管理功能...")
        if hasattr(model, 'task_heads'):
            print(f"   ✓ 模型有task_heads属性，包含 {len(model.task_heads)} 个任务头")
        else:
            print("   ✗ 模型缺少task_heads属性")
            return False
        
        if hasattr(model, 'set_active_head'):
            print("   ✓ 模型有set_active_head方法")
        else:
            print("   ✗ 模型缺少set_active_head方法")
            return False
        
        # 测试设置活动头
        print("4. 测试设置活动头...")
        try:
            model.set_active_head(args.session_name)
            print("   ✓ 成功设置活动头")
        except Exception as e:
            print(f"   ✗ 设置活动头失败: {e}")
            return False
        
        # 检查当前活动头
        print("5. 检查当前活动头...")
        if hasattr(model, 'current_session'):
            print(f"   ✓ 当前活动头: {model.current_session}")
        else:
            print("   ✗ 模型缺少current_session属性")
            return False
        
        # 检查MOE模型是否有fusion_output_dim属性
        print("6. 检查MOE模型属性...")
        if hasattr(model.base_model, 'fusion_output_dim'):
            print(f"   ✓ MOE模型有fusion_output_dim属性: {model.base_model.fusion_output_dim}")
        else:
            print("   ✗ MOE模型缺少fusion_output_dim属性")
            return False
        
        print("\n🎉 所有测试通过！MOE模型与label embedding兼容性良好。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MOE模型与Label Embedding兼容性测试")
    print("=" * 60)
    
    success = test_moe_label_embedding_compat()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 测试通过！MOE模型现在完全支持label embedding功能。")
    else:
        print("❌ 测试失败，需要进一步调试。")
    print("=" * 60) 