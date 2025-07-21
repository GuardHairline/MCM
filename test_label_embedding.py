#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continual.label_embedding import GlobalLabelEmbedding, build_global_label_mapping

def test_get_task_num_labels():
    """测试 get_task_num_labels 方法"""
    print("=== 测试 GlobalLabelEmbedding.get_task_num_labels ===")
    
    # 构建标签映射
    label2idx = build_global_label_mapping()
    print(f"标签映射: {label2idx}")
    
    # 创建 GlobalLabelEmbedding 实例
    label_emb = GlobalLabelEmbedding(label2idx, emb_dim=128)
    
    # 测试每个任务的标签数量
    tasks = ["mabsa", "masc", "mate", "mner"]
    expected_counts = {
        "mabsa": 7,   # O, B-NEG, I-NEG, B-NEU, I-NEU, B-POS, I-POS
        "masc": 3,    # NEG, NEU, POS
        "mate": 3,    # O, B, I
        "mner": 9     # O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
    }
    
    for task in tasks:
        actual_count = label_emb.get_task_num_labels(task)
        expected_count = expected_counts[task]
        print(f"任务 {task}: 实际标签数 {actual_count}, 期望标签数 {expected_count}")
        
        if actual_count != expected_count:
            print(f"❌ 错误: {task} 任务标签数量不匹配!")
            return False
        else:
            print(f"✅ 正确: {task} 任务标签数量匹配")
    
    print("=== 所有测试通过 ===")
    return True

if __name__ == "__main__":
    test_get_task_num_labels() 