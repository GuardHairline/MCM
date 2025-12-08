#!/usr/bin/env python3
"""
生成 Kaggle 全量实验配置文件 (Local Generator)

功能：
1. 生成 60 个独立的 JSON 配置文件 (2 Sequence * 3 Datasets * 10 Strategies)
2. 修复 num_labels 缺失导致的 RuntimeError
3. 生成索引文件以便查询

输出目录: scripts/configs/kaggle_full_experiment/
"""

import json
import os
import sys
from pathlib import Path

# 确保脚本可以导入项目模块 (如果需要)
sys.path.insert(0, str(Path(__file__).parent.parent))

class KaggleConfigGenerator:
    def __init__(self):
        self.output_dir = Path("scripts/configs/kaggle_full_experiment")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义任务的默认标签数量 (修复 RuntimeError: [-1, 768])
        self.task_num_labels = {
            "masc": 3,   # Positive, Neutral, Negative
            "mabsa": 7,  # Positive, Neutral, Negative
            "mate": 3,   # BIO scheme usually or inferred, providing default safe value
            "mner": 9,   # BIO scheme for 4 classes + O, usually inferred but safer to have default
        }

    def get_strategy_args(self, strategy, dataset_name):
        """获取策略特定参数"""
        args = {}
        if strategy == "none": pass
        elif strategy == "DEQA": 
            args["deqa"] = 1
            # DEQA 必须指定 description_file
            if dataset_name == "twitter2015":
                args["description_file"] = "reference/DEQA/DEQA/datasets/release/twitter2015/description_roberta.jsonl"
            elif dataset_name == "twitter2017":
                args["description_file"] = "reference/DEQA/DEQA/datasets/release/twitter2017/description_roberta.jsonl"
            elif dataset_name == "mix":
                args["description_file"] = "reference/DEQA/DEQA/datasets/release/mix/description_roberta.jsonl"
        elif strategy == "moe_adapters": args["moe_adapters"] = 1
        elif strategy == "replay": args["replay"] = 1
        elif strategy == "ewc": args["ewc"] = 1
        elif strategy == "lwf": args["lwf"] = 1
        elif strategy == "mas": args["mas"] = 1
        elif strategy == "si": args["si"] = 1
        elif strategy == "tam_cl": args["tam_cl"] = 1
        elif strategy == "gem": args["gem"] = 1
        return args

    def get_data_path(self, task_name, dataset_name):
        """生成 Kaggle 环境下的数据路径 (/MCM/data/...)"""
        base = "data"
        if task_name == "masc": task_dir = "MASC"
        elif task_name == "mner": task_dir = "MNER"
        elif task_name == "mate": task_dir = "MASC" # 注意代码中通常由 mate 映射到 MNRE 目录
        elif task_name == "mabsa": task_dir = "MASC"
        else: task_dir = "data"
        return f"{base}/{task_dir}/{dataset_name}"

    def generate(self):
        print(f"Generating configs in: {self.output_dir}")
        
        # 定义实验空间
        seq_defs = {
            "seq1": {
                "tasks": ["masc", "mate", "mner", "mabsa", "masc", "mate", "mner", "mabsa"],
                "modes": ["text_only"]*4 + ["multimodal"]*4
            },
            "seq2": {
                "tasks": ["masc", "masc", "mate", "mate", "mner", "mner", "mabsa", "mabsa"],
                "modes": ["text_only", "multimodal"] * 4
            }
        }
        
        # 顺序：15 -> 17 -> mix
        datasets = ["twitter2015", "twitter2017", "mix"]
        # 策略顺序
        strategies = ["none", "DEQA", "moe_adapters", "replay", "ewc", "lwf", "mas", "si", "tam_cl", "gem"]
        
        experiment_index = []
        exp_id = 0
        
        for seq_name, seq_info in seq_defs.items():
            for dataset in datasets:
                for strategy in strategies:
                    
                    # 1. 准备配置元数据
                    config_filename = f"ID{exp_id}_{seq_name}_{dataset}_{strategy}.json"
                    
                    # Kaggle 输出路径固定模式
                    # 所有的输出都会在 /kaggle/working/IDxx_.../ 下
                    output_root = f"/kaggle/working/ID{exp_id}_{seq_name}_{dataset}_{strategy}"
                    train_info_path = f"{output_root}/train_info.json"
                    
                    task_configs = []
                    prev_model_path = ""
                    
                    # 2. 构建 8 个步骤的任务链
                    for step_idx, (task_name, mode) in enumerate(zip(seq_info["tasks"], seq_info["modes"])):
                        
                        current_model_path = f"{output_root}/step{step_idx}_{task_name}_{mode}.pt"
                        data_dir = self.get_data_path(task_name, dataset)

                        task_conf = {
                            "task_name": task_name,
                            "session_name": f"step{step_idx}_{task_name}_{mode}",
                            "head_key": task_name, # 共享 Head
                            "mode": mode,
                            "dataset_name": dataset,
                            "data_dir": data_dir,
                            "train_text_file":f"{data_dir}/train.txt",
                            "test_text_file":f"{data_dir}/test.txt",
                            "dev_text_file":f"{data_dir}/dev.txt",
                            
                            # 模型路径链
                            "pretrained_model_path": prev_model_path,
                            "output_model_path": current_model_path,
                            "train_info_json": train_info_path,
                            
                            # 显式指定 num_labels 避免 -1 错误
                            "num_labels": self.task_num_labels.get(task_name, -1),
                            
                            # 显式指定本地模型路径 (Kaggle 环境下)
                            "text_model_name": "/MCM/downloaded_model/deberta-v3-base",
                            "image_model_name": "/MCM/downloaded_model/vit-base-patch16-224-in21k",
                            
                            # 训练参数
                            "epochs": 20,
                            "batch_size": 8,
                            "lr": 1e-5,
                            "patience": 5,
                            "save_checkpoints": 0,
                            "num_workers": 4, # Kaggle 安全值
                            
                            # 策略参数
                            **self.get_strategy_args(strategy, dataset)
                        }
                        
                        task_configs.append(task_conf)
                        prev_model_path = current_model_path
                    
                    # 3. 组装完整 JSON
                    full_config = {
                        "experiment_id": exp_id,
                        "description": f"{seq_name} on {dataset} using {strategy}",
                        "global_params": {
                            "train_info_json": train_info_path,
                            "output_model_path": f"{output_root}/placeholder.pt", # 必须有，用于 cleanup
                            "dataset_name": dataset,
                            "data_dir": data_dir, 
                            "num_workers": 4,
                            "save_checkpoints": 0,
                            "kaggle_mode": True
                        },
                        "tasks": task_configs
                    }
                    
                    # 4. 保存文件
                    with open(self.output_dir / config_filename, "w", encoding="utf-8") as f:
                        json.dump(full_config, f, indent=2)
                        
                    experiment_index.append({
                        "id": exp_id,
                        "file": config_filename,
                        "seq": seq_name,
                        "dataset": dataset,
                        "strategy": strategy
                    })
                    
                    exp_id += 1
        
        # 保存索引
        with open(self.output_dir / "experiment_index.json", "w", encoding="utf-8") as f:
            json.dump(experiment_index, f, indent=2)
            
        print(f"✅ Generated {exp_id} config files.")
        print(f"📁 Directory: {self.output_dir}")

if __name__ == "__main__":
    generator = KaggleConfigGenerator()
    generator.generate()