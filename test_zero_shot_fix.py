#!/usr/bin/env python3
"""
测试0样本检测修复的脚本
"""

import json
import argparse
import torch
from modules.train_refactored import train
from modules.parser import parse_train_args
from utils.logging import setup_logger

def test_zero_shot_detection():
    """测试0样本检测功能"""
    
    # 创建一个简单的任务配置
    task_config = {
        "tasks": [
            {
                "task_name": "mabsa",
                "session_name": "mabsa_1",
                "train_text_file": "data/MASC/mix/train.txt",
                "test_text_file": "data/MASC/mix/test.txt", 
                "dev_text_file": "data/MASC/mix/dev.txt",
                "image_dir": "data/twitter2015_images",
                "text_model_name": "bert-base-uncased",
                "image_model_name": "resnet50",
                "fusion_strategy": "attention",
                "num_heads": 8,
                "mode": "text_only",
                "hidden_dim": 768,
                "dropout_prob": 0.1,
                "num_labels": 7,
                "epochs": 1,
                "batch_size": 8,
                "lr": 2e-5,
                "weight_decay": 0.01,
                "step_size": 3,
                "gamma": 0.1,
                "patience": 5,
                "strategy": "none",
                "use_label_embedding": False
            },
            {
                "task_name": "mate", 
                "session_name": "mate_2",
                "train_text_file": "data/MASC/mix/train.txt",
                "test_text_file": "data/MASC/mix/test.txt",
                "dev_text_file": "data/MASC/mix/dev.txt", 
                "image_dir": "data/twitter2015_images",
                "text_model_name": "bert-base-uncased",
                "image_model_name": "resnet50",
                "fusion_strategy": "attention",
                "num_heads": 8,
                "mode": "text_only",
                "hidden_dim": 768,
                "dropout_prob": 0.1,
                "num_labels": 3,
                "epochs": 1,
                "batch_size": 8,
                "lr": 2e-5,
                "weight_decay": 0.01,
                "step_size": 3,
                "gamma": 0.1,
                "patience": 5,
                "strategy": "none",
                "use_label_embedding": False
            }
        ],
        "global_params": {
            "data_dir": "data",
            "dataset_name": "twitter2015",
            "num_workers": 2,
            "ewc_dir": "ewc_params",
            "gem_mem_dir": "gem_memory",
            "train_info_json": "test_train_info.json",
            "output_model_path": "test_model.pt"
        }
    }
    
    # 保存任务配置
    config_path = "scripts/configs/test_zero_shot_config.json"
    with open(config_path, 'w') as f:
        json.dump(task_config, f, indent=2)
    
    print(f"任务配置已保存到: {config_path}")
    
    # 创建参数对象
    args = argparse.Namespace()
    
    # 基本参数
    args.task_name = "mabsa"
    args.session_name = "mabsa_1"
    args.task_config_file = config_path
    args.train_info_json = "test_train_info.json"
    args.output_model_path = "test_model.pt"
    
    # 数据参数
    args.data_dir = "data"
    args.dataset_name = "twitter2015"
    args.train_text_file = "data/MASC/mix/train.txt"
    args.test_text_file = "data/MASC/mix/test.txt"
    args.dev_text_file = "data/MASC/mix/dev.txt"
    args.image_dir = "data/twitter2015_images"
    
    # 模型参数
    args.text_model_name = "bert-base-uncased"
    args.image_model_name = "resnet50"
    args.fusion_strategy = "attention"
    args.num_heads = 8
    args.mode = "text_only"
    args.hidden_dim = 768
    args.dropout_prob = 0.1
    args.num_labels = 7
    
    # 训练参数
    args.epochs = 1
    args.batch_size = 8
    args.lr = 2e-5
    args.weight_decay = 0.01
    args.step_size = 3
    args.gamma = 0.1
    args.patience = 5
    args.num_workers = 2
    
    # 持续学习策略参数
    for key in ["ewc", "ewc_lambda", "replay", "memory_percentage", "replay_ratio", 
                "replay_frequency", "lwf", "lwf_T", "lwf_alpha", "lwf_decay",
                "si", "si_epsilon", "si_decay", "mas", "mas_eps", "gem", "gem_mem",
                "pnn", "tam_cl", "moe_adapters", "moe_num_experts", "moe_top_k",
                "ddas", "ddas_threshold", "clap4clip", "mymethod"]:
        setattr(args, key, 0)
    
    # 标签嵌入参数
    args.use_label_embedding = False
    args.label_emb_dim = 128
    args.use_similarity_reg = True
    args.similarity_weight = 0.1
    args.label_embedding_path = None
    
    # 模型头部参数
    args.triaffine = 1
    args.span_hidden = 256
    
    # 图平滑参数
    args.graph_smooth = 1
    args.graph_tau = 0.5
    
    # 目录参数
    args.ewc_dir = "ewc_params"
    args.gem_mem_dir = "gem_memory"
    
    # 日志参数
    args.log_file = None
    
    print("开始测试0样本检测...")
    print(f"任务: {args.task_name}")
    print(f"Session: {args.session_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # 设置日志
    logger = setup_logger(args=args)
    
    try:
        # 开始训练
        best_metrics = train(args, logger)
        print(f"训练完成，最佳指标: {best_metrics}")
        
        # 检查训练信息文件
        import os
        if os.path.exists(args.train_info_json):
            with open(args.train_info_json, 'r') as f:
                train_info = json.load(f)
            
            # 检查是否有0样本指标
            if train_info.get("sessions"):
                last_session = train_info["sessions"][-1]
                if "zero_shot_metrics" in last_session:
                    print("0样本检测指标:")
                    for task, metrics in last_session["zero_shot_metrics"].items():
                        if metrics:
                            print(f"  {task}: {metrics.get('acc', 'N/A'):.2f}%")
                        else:
                            print(f"  {task}: Failed")
                else:
                    print("未找到0样本检测指标")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_zero_shot_detection() 