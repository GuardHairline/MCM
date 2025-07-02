#!/usr/bin/env python3
# scripts/generate_all_scripts.py
"""
批量生成所有常用训练脚本
"""

import os
import subprocess
from config_templates import ConfigTemplate


def generate_common_scripts():
    """
    生成所有常用训练脚本，覆盖本地/服务器、所有策略、所有模式
    """
    # 策略与模式
    strategies = [
        "none", "ewc", "replay", "lwf", "si", "mas", "mymethod",
        "moe", "clap4clip", "labelembedding", "moe_labelembedding", "clap4clip_labelembedding"
    ]
    modes = ["text", "multi", "text2multi"]
    # 本地配置
    local_datasets = ["200"]
    local_epochs = 5
    local_lr = 5e-5
    # 服务器配置
    server_datasets = ["twitter2015", "twitter2017", "mix"]
    server_epochs = 20
    server_lr = 2e-5
    configs = []
    def parse_strategy_flags(strategy):
        return {
            "use_label_embedding": "labelembedding" in strategy,
            "use_moe_adapters": "moe" in strategy,
            "use_clap4clip": "clap4clip" in strategy
        }
    # 本地 AllTask
    for strategy in strategies:
        for mode in modes:
            flags = parse_strategy_flags(strategy)
            configs.append({
                "env": "local",
                "task_type": "AllTask",
                "dataset": "200",
                "strategy": strategy,
                "mode": mode,
                "epochs": local_epochs,
                "lr": local_lr,
                **flags
            })
    # 服务器 AllTask
    for dataset in server_datasets:
        for strategy in strategies:
            for mode in modes:
                flags = parse_strategy_flags(strategy)
                configs.append({
                    "env": "server",
                    "task_type": "AllTask",
                    "dataset": dataset,
                    "strategy": strategy,
                    "mode": mode,
                    "epochs": server_epochs,
                    "lr": server_lr,
                    **flags
                })
    config = ConfigTemplate()
    generated_files = []
    print("开始生成训练脚本...")
    for i, config_dict in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] 生成脚本: {config_dict}")
        script_content = config.generate_script(
            env=config_dict["env"],
            task_type=config_dict["task_type"],
            dataset=config_dict["dataset"],
            strategy=config_dict["strategy"],
            mode=config_dict["mode"],
            use_label_embedding=config_dict.get("use_label_embedding", False),
            use_moe_adapters=config_dict.get("use_moe_adapters", False),
            use_clap4clip=config_dict.get("use_clap4clip", False)
        )
        script_name = config.generate_script_name(
            env=config_dict["env"],
            task_type=config_dict["task_type"],
            dataset=config_dict["dataset"],
            strategy=config_dict["strategy"],
            mode=config_dict["mode"],
            use_label_embedding=config_dict.get("use_label_embedding", False)
        )
        prefix = "local" if config_dict["env"] == "local" else "server"
        script_path = f"scripts/{prefix}/{script_name}"
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        generated_files.append(script_path)
        print(f"  ✓ 已生成: {script_path}")
    print(f"\n总共生成了 {len(generated_files)} 个脚本文件")
    return generated_files


def create_script_index():
    """创建脚本索引文件"""
    index_content = """# 训练脚本索引

## 数据集说明
- **MABSA/MASC/MATE任务**: 共享同一个数据集 (MASC目录下的Twitter2015/Twitter2017)
- **MNER任务**: 使用单独的数据集 (MNER目录下的Twitter2015/Twitter2017)

## 服务器版本 (strain_*)
服务器版本的所有模型都保存为 `1.pt`，文件存储在 `/root/autodl-tmp/` 目录下。

### 多任务训练
- `strain_AllTask_twitter2015_none_multi.sh` - 无持续学习策略
- `strain_AllTask_twitter2015_ewc_multi.sh` - EWC策略
- `strain_AllTask_twitter2015_replay_multi.sh` - Experience Replay
- `strain_AllTask_twitter2015_lwf_multi.sh` - Learning without Forgetting
- `strain_AllTask_twitter2015_si_multi.sh` - Synaptic Intelligence
- `strain_AllTask_twitter2015_mas_multi.sh` - Memory Aware Synapses
- `strain_AllTask_twitter2015_mymethod_multi.sh` - 自定义方法
- `strain_AllTask_twitter2015_moe_multi.sh` - MoE Adapters

### Twitter2017数据集
- `strain_AllTask_twitter2017_none_multi.sh` - 无持续学习
- `strain_AllTask_twitter2017_ewc_multi.sh` - EWC策略
- `strain_AllTask_twitter2017_replay_multi.sh` - Experience Replay

### 单任务训练
- `strain_SingleTask_mabsa_twitter2015_none_multi.sh` - MABSA单任务
- `strain_SingleTask_masc_twitter2015_none_multi.sh` - MASC单任务
- `strain_SingleTask_mate_twitter2015_none_multi.sh` - MATE单任务
- `strain_SingleTask_mner_twitter2015_none_multi.sh` - MNER单任务

### 标签嵌入版本
- `strain_AllTask_twitter2015_none_multi.sh` - 无策略 + 标签嵌入
- `strain_AllTask_twitter2015_ewc_multi.sh` - EWC + 标签嵌入

## 本地版本 (train_*)
本地版本使用详细命名，文件存储在当前目录下。

### 简化数据集 (200样本)
- `train_AllTask_200_none_multi.sh` - 无持续学习
- `train_AllTask_200_ewc_multi.sh` - EWC策略
- `train_AllTask_200_replay_multi.sh` - Experience Replay
- `train_AllTask_200_lwf_multi.sh` - Learning without Forgetting
- `train_AllTask_200_si_multi.sh` - Synaptic Intelligence
- `train_AllTask_200_mas_multi.sh` - Memory Aware Synapses
- `train_AllTask_200_mymethod_multi.sh` - 自定义方法

### 完整数据集
- `train_AllTask_twitter2015_none_multi.sh` - 无持续学习
- `train_AllTask_twitter2015_ewc_multi.sh` - EWC策略
- `train_AllTask_twitter2015_replay_multi.sh` - Experience Replay

### 单任务训练
- `train_SingleTask_mabsa_200_none_multi.sh` - MABSA单任务(简化数据集)
- `train_SingleTask_masc_200_none_multi.sh` - MASC单任务(简化数据集)
- `train_SingleTask_mate_200_none_multi.sh` - MATE单任务(简化数据集)
- `train_SingleTask_mner_200_none_multi.sh` - MNER单任务(简化数据集)
"""

if __name__ == "__main__":
    generate_common_scripts()
    create_script_index()