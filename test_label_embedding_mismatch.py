#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from continual.label_embedding import GlobalLabelEmbedding, build_global_label_mapping
from models.task_heads.token_label_heads import TokenLabelHead

def test_label_embedding_mismatch():
    """测试标签嵌入数量不匹配的问题"""
    print("=== 测试标签嵌入数量不匹配问题 ===")
    
    # 构建标签映射
    label2idx = build_global_label_mapping()
    print(f"全局标签映射: {label2idx}")
    
    # 创建标签嵌入
    label_emb = GlobalLabelEmbedding(label2idx, emb_dim=128)
    
    # 测试参数
    input_dim = 768
    hidden_dim = 256
    batch_size = 2
    seq_len = 8
    
    # 测试 MABSA 任务
    task_name = "mabsa"
    print(f"\n测试任务: {task_name}")
    
    # 获取任务标签数量
    task_num_labels = label_emb.get_task_num_labels(task_name)
    print(f"任务标签数量: {task_num_labels}")
    
    # 获取任务标签嵌入
    task_label_embeddings = label_emb.get_all_label_embeddings(task_name)
    print(f"任务标签嵌入形状: {task_label_embeddings.shape}")
    
    # 创建 TokenLabelHead
    head = TokenLabelHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_labels=task_num_labels,
        label_emb=label_emb,
        task_name=task_name
    )
    
    # 创建输入数据
    seq_feats = torch.randn(batch_size, seq_len, input_dim)
    output = head(seq_feats)
    print(f"输出形状: {output.shape}")
    assert output.shape == (batch_size, seq_len, task_num_labels), f"输出形状不匹配: {output.shape}"
    
    print("✅ 所有测试通过")
    return True

if __name__ == "__main__":
    test_label_embedding_mismatch() 