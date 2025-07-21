#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

def test_sequence_loss():
    """测试序列标注任务的损失计算"""
    print("=== 测试序列标注任务损失计算 ===")
    
    # 模拟 BiaffineSpanHead 的输出
    batch_size = 2
    seq_len = 10
    num_labels = 7
    
    # 创建 logits (batch_size, seq_len, seq_len, num_labels)
    logits = torch.randn(batch_size, seq_len, seq_len, num_labels)
    
    # 创建标签 (batch_size, seq_len)
    labels = torch.randint(0, num_labels, (batch_size, seq_len))
    
    print(f"原始 logits 形状: {logits.shape}")
    print(f"原始 labels 形状: {labels.shape}")
    
    try:
        # 提取对角线元素
        if logits.dim() == 4 and logits.size(1) == logits.size(2):
            logits_diag = logits.diagonal(dim1=1, dim2=2)
            print(f"对角线 logits 形状: {logits_diag.shape}")
        
        # 计算损失
        loss = F.cross_entropy(
            logits_diag.reshape(-1, logits_diag.size(-1)),
            labels.reshape(-1),
            ignore_index=-100
        )
        
        print(f"✅ 序列标注损失计算成功: {loss.item():.4f}")
        
        # 验证形状匹配
        expected_logits_shape = (batch_size * seq_len, num_labels)
        expected_labels_shape = (batch_size * seq_len,)
        
        actual_logits_shape = logits_diag.reshape(-1, logits_diag.size(-1)).shape
        actual_labels_shape = labels.reshape(-1).shape
        
        print(f"期望 logits 形状: {expected_logits_shape}")
        print(f"实际 logits 形状: {actual_logits_shape}")
        print(f"期望 labels 形状: {expected_labels_shape}")
        print(f"实际 labels 形状: {actual_labels_shape}")
        
        assert actual_logits_shape == expected_logits_shape, f"logits 形状不匹配: {actual_logits_shape} vs {expected_logits_shape}"
        assert actual_labels_shape == expected_labels_shape, f"labels 形状不匹配: {actual_labels_shape} vs {expected_labels_shape}"
        
        print("✅ 形状匹配验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_sequence_loss() 