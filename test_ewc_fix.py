#!/usr/bin/env python3
"""
测试EWC修复的脚本
"""

import json
import argparse
import torch
from modules.train_refactored import train
from modules.parser import parse_train_args
from utils.logging import setup_logger

def test_ewc_fix():
    """测试EWC修复"""
    
    # 创建一个简单的任务配置
    task_config = {
        "tasks": [
            {
                "task_name": "masc",
                "session_name": "masc_1",
                "train_text_file": "data/MASC/mix/train.txt",
                "test_text_file": "data/MASC/mix/test.txt", 
                "dev_text_file": "data/MASC/mix/dev.txt",
                "image_dir": "data/twitter2015_images",
                "text_model_name": "bert-base-uncased",
                "image_model_name": "resnet50",
                "fusion_strategy": "attention",
                "mode": "text_only",
                "batch_size": 16,
                "epochs": 1,
                "learning_rate": 2e-5,
                "max_length": 128,
                "num_workers": 0,
                "device": "cpu",
                "ewc": 1,
                "ewc_lambda": 1000.0
            }
        ]
    }
    
    print("=== 测试EWC修复 ===")
    print("任务配置:")
    for i, task in enumerate(task_config["tasks"]):
        print(f"  任务{i+1}: task_name={task['task_name']}, session_name={task['session_name']}, ewc={task.get('ewc', 0)}")
    
    # 保存测试配置
    with open("test_ewc_config.json", "w") as f:
        json.dump(task_config, f, indent=2)
    
    print("\n测试配置文件已保存到: test_ewc_config.json")
    print("你可以使用以下命令测试：")
    print("python scripts/train_with_zero_shot.py --task_config test_ewc_config.json --task_idx 0")

if __name__ == "__main__":
    test_ewc_fix() 