#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动修复Twitter2015配置文件的问题

修复内容:
1. DEQA: 修正mner和mabsa的num_labels
2. MoE/None: 添加global_params
3. 统一image_dir路径
4. 统一epochs为20
5. 修正train_info命名
"""
import json
import os
from pathlib import Path

# 配置文件路径
CONFIG_DIR = Path("scripts/configs")
CONFIGS = [
    "server_twitter2015_deqa_seq1.json",
    "server_twitter2015_moe_seq1.json",
    "server_twitter2015_none_seq1.json"
]

def fix_deqa_config(config):
    """修复DEQA配置"""
    print("修复DEQA配置...")
    
    # 1. 修正global_params中的命名
    if "global_params" in config:
        config["global_params"]["train_info_json"] = "checkpoints/train_info_twitter2015_deqa_t2m.json"
        config["global_params"]["output_model_path"] = "checkpoints/model_twitter2015_deqa_t2m.pt"
    
    # 2. 修正所有任务
    for i, task in enumerate(config["tasks"]):
        # 修正num_labels
        if task["task_name"] == "mner":
            task["num_labels"] = 9
            print(f"  Task {i+1} (mner): num_labels 改为 9")
        elif task["task_name"] == "mabsa":
            task["num_labels"] = 7
            print(f"  Task {i+1} (mabsa): num_labels 改为 7")
        
        # 统一image_dir
        task["image_dir"] = "data/twitter2015_images"
        
        # 统一epochs为20
        if task["epochs"] != 20:
            print(f"  Task {i+1} ({task['task_name']}): epochs {task['epochs']} → 20")
            task["epochs"] = 20
    
    return config

def fix_moe_config(config):
    """修复MoE配置"""
    print("修复MoE配置...")
    
    # 1. 添加global_params
    if "global_params" not in config:
        print("  添加global_params")
        config["global_params"] = {
            "train_info_json": "checkpoints/train_info_twitter2015_moe_t2m.json",
            "output_model_path": "checkpoints/model_twitter2015_moe_t2m.pt",
            "ewc_dir": "ewc_params",
            "gem_mem_dir": "gem_memory",
            "data_dir": "data",
            "dataset_name": "twitter2015",
            "num_workers": 4
        }
    
    # 2. 修正所有任务
    for i, task in enumerate(config["tasks"]):
        # 统一image_dir
        if task["image_dir"] != "data/twitter2015_images":
            task["image_dir"] = "data/twitter2015_images"
            print(f"  Task {i+1} ({task['task_name']}): image_dir 改为 data/twitter2015_images")
        
        # 统一epochs为20
        if task["epochs"] != 20:
            print(f"  Task {i+1} ({task['task_name']}): epochs {task['epochs']} → 20")
            task["epochs"] = 20
        
        # 添加缺失的MoE参数
        if "moe_balance_coef" not in task:
            task["moe_balance_coef"] = 0.01
            print(f"  Task {i+1} ({task['task_name']}): 添加 moe_balance_coef=0.01")
        
        if "lora_rank" not in task:
            task["lora_rank"] = 8
            print(f"  Task {i+1} ({task['task_name']}): 添加 lora_rank=8")
        
        if "moe_expert_type" not in task:
            task["moe_expert_type"] = "lora"
            print(f"  Task {i+1} ({task['task_name']}): 添加 moe_expert_type=lora")
    
    return config

def fix_none_config(config):
    """修复None配置"""
    print("修复None配置...")
    
    # 1. 添加global_params
    if "global_params" not in config:
        print("  添加global_params")
        config["global_params"] = {
            "train_info_json": "checkpoints/train_info_twitter2015_none_t2m.json",
            "output_model_path": "checkpoints/model_twitter2015_none_t2m.pt",
            "ewc_dir": "ewc_params",
            "gem_mem_dir": "gem_memory",
            "data_dir": "data",
            "dataset_name": "twitter2015",
            "num_workers": 4
        }
    
    # 2. 修正所有任务
    for i, task in enumerate(config["tasks"]):
        # 统一image_dir
        if task["image_dir"] != "data/twitter2015_images":
            task["image_dir"] = "data/twitter2015_images"
            print(f"  Task {i+1} ({task['task_name']}): image_dir 改为 data/twitter2015_images")
        
        # 统一epochs为20
        if task["epochs"] != 20:
            print(f"  Task {i+1} ({task['task_name']}): epochs {task['epochs']} → 20")
            task["epochs"] = 20
    
    return config

def backup_config(config_path):
    """备份原配置文件"""
    backup_path = config_path.with_suffix(".json.backup")
    if not backup_path.exists():
        import shutil
        shutil.copy(config_path, backup_path)
        print(f"✓ 已备份原配置: {backup_path.name}")
    else:
        print(f"⚠ 备份已存在，跳过: {backup_path.name}")

def main():
    print("="*80)
    print("Twitter2015配置文件自动修复工具")
    print("="*80)
    print()
    
    for config_name in CONFIGS:
        config_path = CONFIG_DIR / config_name
        
        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"处理配置文件: {config_name}")
        print(f"{'='*80}")
        
        # 备份原文件
        backup_config(config_path)
        
        # 读取配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 根据策略类型修复
        strategy = config.get("strategy", "")
        if strategy == "deqa":
            config = fix_deqa_config(config)
        elif strategy == "moe":
            config = fix_moe_config(config)
        elif strategy == "none":
            config = fix_none_config(config)
        else:
            print(f"⚠ 未知策略: {strategy}")
            continue
        
        # 保存修复后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 配置文件已修复并保存: {config_name}")
    
    print("\n" + "="*80)
    print("✓ 所有配置文件修复完成！")
    print("="*80)
    print("\n备注:")
    print("  - 原配置已备份为 *.json.backup")
    print("  - 可以查看 CONFIG_CHECK_REPORT.md 了解详细修改")
    print("  - 运行前请确认修改正确")

if __name__ == "__main__":
    main()






