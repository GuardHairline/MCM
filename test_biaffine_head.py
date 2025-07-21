#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from continual.label_embedding import GlobalLabelEmbedding, build_global_label_mapping
from models.task_heads.token_label_heads import TokenLabelHead

def test_token_label_head():
    """测试 TokenLabelHead 是否能正常工作"""
    print("=== 测试 TokenLabelHead ===")
    
    # 构建标签映射
    label2idx = build_global_label_mapping()
    
    # 创建标签嵌入
    label_emb = GlobalLabelEmbedding(label2idx, emb_dim=128)
    
    # 测试参数
    input_dim = 768
    hidden_dim = 256
    batch_size = 2
    seq_len = 10
    num_labels = 7
    
    head = TokenLabelHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_labels=num_labels,
        label_emb=label_emb,
        task_name="mabsa"
    )
    
    # 创建输入数据
    seq_feats = torch.randn(batch_size, seq_len, input_dim)
    try:
        output = head(seq_feats)
        print(f"✅ TokenLabelHead 测试通过，输出形状: {output.shape}")
        expected_shape = (batch_size, seq_len, num_labels)
        assert output.shape == expected_shape, f"输出形状不匹配: {output.shape} vs {expected_shape}"
    except Exception as e:
        print(f"❌ TokenLabelHead 测试失败: {e}")
        return False
    return True

if __name__ == "__main__":
    test_token_label_head() 