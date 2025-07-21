#!/usr/bin/env python3
# test_session_name.py
"""
测试session_name参数传递
"""

import sys
import os
import torch
import torch.nn as nn
from models.task_heads.sent_label_attn import LabelAttentionSentHead
from models.task_heads.token_label_heads import TokenLabelHead
from continual.label_embedding import GlobalLabelEmbedding, build_global_label_mapping
from modules.train_utils import Full_Model
from models.base_model import BaseMultimodalModel

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.parser import parse_train_args

def test_session_name():
    """测试session_name参数"""
    # 模拟命令行参数
    sys.argv = [
        'test_session_name.py',
        '--task_name', 'mabsa',
        '--session_name', 'session_1_mabsa_200_none',
        '--train_info_json', 'test_train_info.json',
        '--output_model_path', 'test_model.pt',
        '--data_dir', './data',
        '--dataset_name', '200',
        '--train_text_file', 'data/MASC/twitter2015/train__.txt',
        '--dev_text_file', 'data/MASC/twitter2015/dev__.txt',
        '--test_text_file', 'data/MASC/twitter2015/test__.txt',
        '--num_labels', '7',
        '--mode', 'text_only',
        '--epochs', '1'
    ]
    
    args = parse_train_args()
    print(f"Parsed session_name: '{args.session_name}'")
    print(f"Session name length: {len(args.session_name)}")
    print(f"Session name bytes: {args.session_name.encode('utf-8')}")
    
    # 测试create_session_info
    from modules.train_utils import create_session_info
    session_info = create_session_info(args)
    print(f"Session info session_name: '{session_info['session_name']}'")
    print(f"Session info session_name length: {len(session_info['session_name'])}")

def test_full_model_set_active_head():
    print("=== 测试 Full_Model 的 set_active_head 方法 ===")
    
    # 创建标签嵌入
    label2idx = build_global_label_mapping()
    label_emb = GlobalLabelEmbedding(label2idx, emb_dim=128)
    
    # 创建基础模型 - 使用 text_only 模式避免图像模型加载
    base_model = BaseMultimodalModel(
        text_model_name="microsoft/deberta-v3-base",
        image_model_name="google/vit-base-patch16-224-in21k",
        multimodal_fusion="concat",
        num_heads=8,
        mode="text_only"  # 只使用文本模态，避免图像模型加载
    )
    
    # 创建 MASC 任务的模型头
    masc_head = LabelAttentionSentHead(
        input_dim=768,
        num_labels=3,
        label_emb=label_emb,
        task_name="masc"
    )
    
    # 创建 MABSA 任务的模型头
    mabsa_head = TokenLabelHead(
        input_dim=768,
        hidden_dim=256,
        num_labels=7,
        label_emb=label_emb,
        task_name="mabsa"
    )
    
    # 创建 Full_Model
    full_model = Full_Model(base_model, masc_head, dropout_prob=0.1)
    
    # 添加任务头
    full_model.add_task_head("session_masc", "masc", masc_head, None)
    full_model.add_task_head("session_mabsa", "mabsa", mabsa_head, None)
    
    print(f"初始模型头类型: {type(full_model.head).__name__}")
    print(f"当前会话: {full_model.current_session}")
    
    # 测试设置活动头
    print("\n--- 测试设置 MASC 任务头 ---")
    full_model.set_active_head("session_masc")
    print(f"设置后的模型头类型: {type(full_model.head).__name__}")
    print(f"当前会话: {full_model.current_session}")
    print(f"当前任务名称: {full_model.get_current_task_name()}")
    
    print("\n--- 测试设置 MABSA 任务头 ---")
    full_model.set_active_head("session_mabsa")
    print(f"设置后的模型头类型: {type(full_model.head).__name__}")
    print(f"当前会话: {full_model.current_session}")
    print(f"当前任务名称: {full_model.get_current_task_name()}")
    
    # 测试错误情况
    print("\n--- 测试错误情况 ---")
    try:
        full_model.set_active_head("nonexistent_session")
    except ValueError as e:
        print(f"预期的错误: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_session_name()
    test_full_model_set_active_head() 