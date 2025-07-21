#!/usr/bin/env python3
"""
测试使用session_name作为唯一标识的0样本检测功能
"""

import json
import argparse
import torch
from modules.train_refactored import train
from modules.parser import parse_train_args
from utils.logging import setup_logger

def test_session_name_zero_shot():
    """测试使用session_name的0样本检测功能"""
    
    # 创建一个包含重复task_name但不同session_name的任务配置
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
                "device": "cpu"
            },
            {
                "task_name": "masc",  # 相同的task_name
                "session_name": "masc_2",  # 不同的session_name
                "train_text_file": "data/MASC/mix/train.txt",
                "test_text_file": "data/MASC/mix/test.txt", 
                "dev_text_file": "data/MASC/mix/dev.txt",
                "image_dir": "data/twitter2015_images",
                "text_model_name": "bert-base-uncased",
                "image_model_name": "resnet50",
                "fusion_strategy": "attention",
                "mode": "multimodal",  # 不同的模式
                "batch_size": 16,
                "epochs": 1,
                "learning_rate": 2e-5,
                "max_length": 128,
                "num_workers": 0,
                "device": "cpu"
            },
            {
                "task_name": "mate",
                "session_name": "mate_1",
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
                "device": "cpu"
            }
        ]
    }
    
    print("=== 测试使用session_name的0样本检测 ===")
    print("任务配置：")
    for i, task in enumerate(task_config["tasks"]):
        print(f"  任务{i+1}: task_name={task['task_name']}, session_name={task['session_name']}, mode={task['mode']}")
    
    print("\n预期行为：")
    print("1. 训练masc_1后，应该对masc_2和mate_1进行0样本检测")
    print("2. 训练masc_2后，应该对mate_1进行0样本检测")
    print("3. 每个session_name都应该被独立检测")
    
    # 保存测试配置
    with open("test_session_config.json", "w") as f:
        json.dump(task_config, f, indent=2)
    
    print("\n测试配置文件已保存到: test_session_config.json")
    print("你可以使用以下命令测试：")
    print("python scripts/train_with_zero_shot.py --task_config test_session_config.json --task_idx 0")

if __name__ == "__main__":
    test_session_name_zero_shot() 