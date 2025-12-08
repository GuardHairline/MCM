#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from scripts.generate_task_config import TaskConfigGenerator
except ImportError:
    class TaskConfigGenerator:
        def __init__(self): self.environments = {}

class AutoDLFullExperimentGenerator(TaskConfigGenerator):
    def __init__(self):
        super().__init__()
        from datetime import datetime
        date_folder = datetime.now().strftime("%y%m%d")
        self.root_dir = f"/root/autodl-tmp/experiments/{date_folder}"
        
        self.environments["autodl"] = {
            "data_dir": "data",
            "image_dir": "data/img",
            "model_root": self.root_dir
        }

    def get_strategy_args(self, strategy, dataset_name):
        """根据策略和数据集生成特定参数"""
        args = {}
        if strategy == "none": pass
        elif strategy == "DEQA": 
            args["deqa"] = 1
            # DEQA 专用路径
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

    def generate(self):
        output_dir = Path("scripts/configs/autodl_full")
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        datasets = ["twitter2015", "twitter2017", "mix"]
        strategies = ["none", "DEQA", "moe_adapters", "replay", "ewc", "lwf", "mas", "si", "tam_cl", "gem"]
        
        index = []
        experiment_id = 0
        
        for seq_name, seq_info in seq_defs.items():
            for dataset in datasets:
                for strategy in strategies:
                    
                    exp_name = f"ID{experiment_id:02d}_{seq_name}_{dataset}_{strategy}"
                    exp_dir = f"{self.root_dir}/{exp_name}"
                    train_info_path = f"{exp_dir}/train_info.json"
                    
                    task_configs = []
                    prev_model_path = "" 
                    
                    for step_idx, (task_name, mode) in enumerate(zip(seq_info["tasks"], seq_info["modes"])):
                        head_key = task_name 
                        current_model_path = f"{exp_dir}/step{step_idx}_{task_name}_{mode}.pt"
                        
                        task_conf = {
                            "task_name": task_name,
                            "session_name": f"step{step_idx}_{task_name}_{mode}",
                            "head_key": head_key,
                            "mode": mode,
                            "dataset_name": dataset,
                            "data_dir": f"data/{'MASC' if task_name=='masc' else 'MNER' if task_name=='mner' else 'MNRE' if task_name=='mate' else 'MABSA' if task_name=='mabsa' else 'data'}/{dataset}",
                            "train_text_file": "train.txt",
                            "test_text_file": "test.txt",
                            "dev_text_file": "dev.txt",
                            
                            # 显式指定本地模型路径 (假设 AutoDL 上也解压到了根目录)
                            "text_model_name": "downloaded_model/deberta-v3-base",
                            "image_model_name": "downloaded_model/vit-base-patch16-224-in21k",
                            
                            "pretrained_model_path": prev_model_path,
                            "output_model_path": current_model_path,
                            "train_info_json": train_info_path,
                            "epochs": 20,
                            "batch_size": 16, 
                            "lr": 1e-5,
                            "patience": 5,
                            "save_checkpoints": 0,
                            "num_workers": 4,
                            # 传入 dataset_name 获取 DEQA 参数
                            **self.get_strategy_args(strategy, dataset)
                        }
                        
                        prev_model_path = current_model_path
                        task_configs.append(task_conf)
                    
                    full_config = {
                        "experiment_id": experiment_id,
                        "description": f"{seq_name} on {dataset} using {strategy}",
                        "global_params": {
                            "train_info_json": train_info_path,
                            "output_model_path": f"{exp_dir}/placeholder.pt",
                            "dataset_name": dataset,
                            "data_dir": "data",
                            "num_workers": 4,
                            "save_checkpoints": 0
                        },
                        "tasks": task_configs
                    }
                    
                    config_filename = f"{exp_name}.json"
                    config_path = output_dir / config_filename
                    
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(full_config, f, indent=2)
                        
                    index.append({
                        "id": experiment_id,
                        "name": exp_name,
                        "config_path": str(config_path),
                        "seq": seq_name,
                        "dataset": dataset,
                        "strategy": strategy
                    })
                    
                    experiment_id += 1

        with open(output_dir / "experiment_index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
            
        print(f"✅ 生成了 {len(index)} 个实验配置文件，保存在 {output_dir}")
        
        with open(output_dir / "run_all_autodl.sh", "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n\n")
            f.write("echo '=== AutoDL Full Experiment ==='\n")
            f.write("echo 'Step 0: Installing dependencies...'\n")
            f.write("pip install -r requirements.txt\n\n")
            f.write("mkdir -p logs\n\n")
            
            for item in index:
                cmd = f"python scripts/train_with_zero_shot.py --config {item['config_path']} > logs/{item['name']}.log 2>&1"
                f.write(f"echo 'Running Experiment {item['id']}: {item['name']}'\n")
                f.write(f"{cmd}\n")
                f.write("if [ $? -ne 0 ]; then\n")
                f.write(f"    echo '❌ Error in experiment {item['id']}'\n")
                f.write("else\n")
                f.write(f"    echo '✅ Finished experiment {item['id']}'\n")
                f.write("fi\n\n")
                
        print(f"✅ 生成了运行脚本: {output_dir}/run_all_autodl.sh")

if __name__ == "__main__":
    generator = AutoDLFullExperimentGenerator()
    generator.generate()