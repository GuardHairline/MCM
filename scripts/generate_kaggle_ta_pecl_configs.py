#!/usr/bin/env python3
"""
生成 Head 策略对比实验配置 (4头 vs 8头) - 修复版 V2
路径: scripts/generate_kaggle_head_comparison.py

修复记录：
1. [Fix] 添加 global_params["output_model_path"] 防止 train_with_zero_shot.py 崩溃。
2. [Fix] 添加 global_params["ewc_dir"] 等默认路径，防止 CL 模块检查报错。
"""

import json
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from scripts.generate_task_config import TaskConfigGenerator

def generate_configs():
    # ==========================================
    # 1. 路径配置
    # ==========================================
    # 本地保存配置文件的路径
    current_dir = Path(__file__).parent
    config_save_dir = current_dir / "configs" / "kaggle_ta_pecl"
    config_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Kaggle 运行时环境路径
    KAGGLE_ROOT = "/MCM"                # 代码和数据根目录
    KAGGLE_OUTPUT = "/kaggle/working"   # 输出根目录
    DATA_DIR = f"{KAGGLE_ROOT}/data"    # 数据目录
    
    print(f"配置文件将保存到: {config_save_dir}")
    
    generator = TaskConfigGenerator()
    
    # ==========================================
    # 2. 实验序列定义 (8步)
    # ==========================================
    task_sequence = ["masc", "mate", "mner", "mabsa", "masc", "mate", "mner", "mabsa"]
    mode_sequence = ["text_only"] * 4 + ["multimodal"] * 4
    dataset = "twitter2015"
    
    # ==========================================
    # 3. 通用参数
    # ==========================================
    common_params = {
        "use_crf": 1,
        "use_bilstm": 0,
        "use_span_loss": 0,
        "triaffine": 0,
        "lr": 1e-3,
        "batch_size": 16,
        "epochs": 20,
        "patience": 5,
        "save_checkpoints": 0,
        "debug_samples": 100,
        "num_workers": 4,
        "data_dir": DATA_DIR,
        "image_dir": f"{DATA_DIR}/img",
        "ta_pecl": 1,
        "ta_pecl_top_k": 4,
        "description_file": "reference/DEQA/DEQA/datasets/release/twitter2015/description_roberta.jsonl"
    }

    # ==========================================
    # 4. 生成函数
    # ==========================================
    def create_experiment_config(exp_name, head_keys, subfolder_name):
        # 实验输出目录 (在 /kaggle/working 下)
        exp_output_dir = f"{KAGGLE_OUTPUT}/output/{subfolder_name}"
        
        # 全局文件路径
        train_info_path = f"{exp_output_dir}/train_info.json"
        # 关键修复：定义一个全局模型路径 (虽然每个Task有自己的，但脚本需要这个key)
        final_output_path = f"{exp_output_dir}/model.pt"
        
        tasks_config = []
        prev_model_path = "" 
        
        for i, (task_name, mode) in enumerate(zip(task_sequence, mode_sequence)):
            session_id = i + 1
            step_name = f"step{session_id}_{task_name}_{mode}"
            current_output_model = f"{exp_output_dir}/model.pt"
            
            task_conf = generator.create_task_config(
                task_name=task_name,
                session_name=step_name,
                dataset=dataset,
                env="kaggle", 
                strategy="ta_pecl",
                mode=mode,
                **common_params
            )
            
            # --- 定制参数 ---
            task_conf["head_key"] = head_keys[i]
            task_conf["save_checkpoints"] = common_params.get("save_checkpoints", 0)
            task_conf["debug_samples"] = common_params.get("debug_samples", 100)
            task_conf["data_dir"] = common_params.get("data_dir", DATA_DIR)
            task_conf["ta_pecl"] = common_params.get("ta_pecl", 1)
            task_conf["ta_pecl_top_k"] = common_params.get("ta_pecl_top_k", 4)
            # 链式加载
            if i > 0:
                task_conf["pretrained_model_path"] = prev_model_path
            else:
                task_conf["pretrained_model_path"] = ""
            
            task_conf["output_model_path"] = current_output_model
            
            # 修正数据路径
            for key in ["train_text_file", "dev_text_file", "test_text_file"]:
                if task_conf.get(key) and not task_conf[key].startswith("/"):
                    task_conf[key] = f"{KAGGLE_ROOT}/{task_conf[key]}"
            
            tasks_config.append(task_conf)
            prev_model_path = current_output_model

        # 组装完整 JSON
        full_config = {
            "experiment_name": exp_name,
            "global_params": {
                # [关键修复] 必须包含 output_model_path
                "output_model_path": final_output_path, 
                "train_info_json": train_info_path,
                "output_root": exp_output_dir,
                "dataset_root": DATA_DIR,
                "log_file": f"{exp_output_dir}/{exp_name}.log",
                "save_checkpoints": 0,
                
                # [防崩溃] 添加 CL 相关路径占位符
                "ewc_dir": f"{exp_output_dir}/ewc",
                "gem_mem_dir": f"{exp_output_dir}/gem",
                "checkpoint_dir": exp_output_dir
            },
            "tasks": tasks_config
        }
        return full_config

    # ==========================================
    # 实验 1: 4-Head
    # ==========================================
    keys_4head = ["masc", "mate", "mner", "mabsa", "masc", "mate", "mner", "mabsa"]
    config_1 = create_experiment_config("TA_PECL", keys_4head, "ta_pecl")
    
    path_1 = config_save_dir / "ta_pecl.json"
    with open(path_1, 'w', encoding='utf-8') as f:
        json.dump(config_1, f, indent=4)
    print(f"✅ Generated: {path_1}")


    # 生成索引
    index = {
        "1": f"scripts/configs/kaggle_ta_pecl/ta_pecl.json",
    }
    with open(config_save_dir / "experiment_index.json", 'w') as f:
        json.dump(index, f, indent=4)
    print(f"✅ Index Generated")

if __name__ == "__main__":
    generate_configs()