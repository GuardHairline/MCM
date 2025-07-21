#!/usr/bin/env python3
"""
简单测试零样本评估修复
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

class DummyLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    def warning(self, msg):
        print(f"[WARNING] {msg}")
    def error(self, msg):
        print(f"[ERROR] {msg}")

def test_zero_shot_fix():
    """测试零样本评估修复"""
    print("开始测试零样本评估修复...")
    
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
            use_label_embedding=True,  # 测试带label embedding
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
        
        # 测试标准模型
        print("\n=== 测试标准模型 ===")
        model = create_model(args, device, logger=logger)
        
        try:
            metrics = evaluate_single_task(model, "masc", "test", device, args)
            print(f"标准模型零样本评估成功: {metrics['acc']:.4f}")
            return True
        except Exception as e:
            print(f"标准模型零样本评估失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_zero_shot_fix()
    if success:
        print("\n测试通过！零样本评估修复成功。")
    else:
        print("\n测试失败，需要进一步调试。") 