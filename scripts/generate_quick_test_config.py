#!/usr/bin/env python3
"""
快速测试配置生成器

生成一个最小化配置用于快速测试项目是否可运行。
- epoch=2
- batch_size=4 (最小)
- 策略=NONE
- 任务序列：MASC→MATE→MNER→MABSA (text-only) → MASC→MATE→MNER→MABSA (multimodal)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def create_quick_test_config():
    """创建快速测试配置"""
    
    # 任务列表 - 使用200小数据集
    tasks = [
        # Text-only 模式
        {"name": "masc", "mode": "text_only", "dataset": "200"},
        {"name": "mate", "mode": "text_only", "dataset": "200"},
        {"name": "mner", "mode": "text_only", "dataset": "200"},
        {"name": "mabsa", "mode": "text_only", "dataset": "200"},
        # Multimodal 模式
        {"name": "masc", "mode": "multimodal", "dataset": "200"},
        {"name": "mate", "mode": "multimodal", "dataset": "200"},
        {"name": "mner", "mode": "multimodal", "dataset": "200"},
        {"name": "mabsa", "mode": "multimodal", "dataset": "200"},
    ]
    
    # 生成任务配置列表
    task_configs = []
    
    for i, task_info in enumerate(tasks):
        task_name = task_info["name"]
        mode = task_info["mode"]
        dataset = task_info["dataset"]
        
        # session名称
        session_name = f"{task_name}_{i+1}"  # 使用索引号，如 masc_1, mate_2
        
        # 数据路径（200数据集使用__后缀）
        if task_name == "mner":
            # MNER任务使用MNER目录
            data_dir = "data/MNER/twitter2015"
            train_file = f"{data_dir}/train__.txt"
            dev_file = f"{data_dir}/dev__.txt"
            test_file = f"{data_dir}/test__.txt"
        else:
            # 其他任务使用MASC目录
            data_dir = "data/MASC/twitter2015"
            train_file = f"{data_dir}/train__.txt"
            dev_file = f"{data_dir}/dev__.txt"
            test_file = f"{data_dir}/test__.txt"
        
        task_config = {
            # ========== 任务信息 ==========
            "task_name": task_name,
            "session_name": session_name,
            "dataset": dataset,
            "mode": mode,
            
            # ========== 数据路径 ==========
            "train_text_file": train_file,
            "dev_text_file": dev_file,
            "test_text_file": test_file,
            "image_dir": "data/img",
            
            # ========== 模型参数 ==========
            "num_labels": {"masc": 3, "mate": 3, "mner": 9, "mabsa": 7}[task_name],
            "text_model_name": "microsoft/deberta-v3-base",
            "image_model_name": "google/vit-base-patch16-224-in21k",
            "fusion_strategy": "concat",
            "hidden_dim": 768,
            "dropout_prob": 0.3,
            "num_heads": 8,
            
            # ========== 训练参数（最快设置）==========
            "epochs": 2,  # 最小epoch
            "batch_size": 4,  # 最小batch size
            "lr": 5e-5,
            "lstm_lr": 1e-4,  # BiLSTM学习率
            "crf_lr": 1e-3,   # CRF学习率
            "weight_decay": 1e-5,
            "step_size": 10,
            "gamma": 0.5,
            "patience": 999,  # 禁用早停
            
            # ========== 持续学习策略 ==========
            "ewc": 0,
            "fisher_selector": 0,
            "replay": 0,
            "lwf": 0,
            "si": 0,
            "mas": 0,
            "gem": 0,
            "agem": 0,
            "moe_adapters": 0,
            "tam_cl": 0,
            "deqa": 0,
            "clap4clip": 0,
            "ddas": 0,
            
            # ========== 模型头部参数 ==========
            "triaffine": 1,
            "span_hidden": 256,
            "use_crf": 1,
            "use_span_loss": 1,
            "boundary_weight": 0.2,
            "span_f1_weight": 0.0,
            "transition_weight": 0.0,
            
            # ========== 图平滑参数 ==========
            "graph_smooth": 1,
            "graph_tau": 0.5,
            
            # ========== 其他 ==========
            "use_label_embedding": 0,
            "use_hierarchical_head": 0,
            "num_workers": 0,
            "description_file": None,
        }
        
        task_configs.append(task_config)
    
    # 创建完整配置（与generate_task_config.py格式一致）
    complete_config = {
        "env": "local",
        "dataset": "200",
        "strategy": "none",
        "mode_sequence": [t["mode"] for t in tasks],
        "mode_suffix": "t2m",  # text_only to multimodal
        "use_label_embedding": False,
        "seq_suffix": "quick_test",
        "total_tasks": len(task_configs),
        "tasks": task_configs,
        "global_params": {
            "base_dir": "./",
            "output_model_path": "checkpoints/quick_test/model_200_none_t2m_quick_test.pt",
            "train_info_json": "checkpoints/quick_test/train_info_200_none_t2m_quick_test.json",
            "task_heads_path": "checkpoints/quick_test/model_200_none_t2m_quick_test_task_heads.pt",
            "label_embedding_path": "checkpoints/quick_test/label_embedding_200_none_t2m_quick_test.pt",
            "ewc_dir": "checkpoints/quick_test/ewc_params",
            "gem_mem_dir": "checkpoints/quick_test/gem_memory",
            "log_dir": "checkpoints/quick_test/logs",
            "checkpoint_dir": "checkpoints/quick_test",
            "num_workers": 0,
            "data_dir": "./data",
            "dataset_name": "200",
        }
    }
    
    return complete_config


def save_configs(config: Dict, output_dir: str):
    """保存配置文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存单个完整配置文件
    config_file = output_path / "quick_test_config.json"
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 生成配置: {config_file}")
    print(f"  - 任务数: {config['total_tasks']}")
    print(f"  - 策略: {config['strategy']}")
    print(f"  - 数据集: {config['dataset']}")
    
    # 生成运行脚本
    generate_run_script(config, output_path)


def generate_run_script(config: Dict, output_path: Path):
    """生成运行脚本"""
    script_path = output_path / "run_quick_test.sh"
    
    total_tasks = config['total_tasks']
    config_file = "scripts/configs/quick_test/quick_test_config.json"
    
    with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f"""#!/bin/bash
#===============================================================================
# 快速测试脚本
# 用途：测试项目是否可以正常运行
# 设置：epoch=2, batch_size=4, 策略=NONE
# 任务：{total_tasks}个任务 (4个text-only + 4个multimodal)
#===============================================================================

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "==============================================================================="
echo "快速测试 - 开始"
echo "==============================================================================="
echo "项目根目录: $PROJECT_ROOT"
echo "配置文件: {config_file}"
echo "任务数: {total_tasks}"
echo ""

# 运行所有任务
echo "==============================================================================="
echo "运行 {total_tasks} 个任务..."
echo "==============================================================================="
python -m scripts.train_with_zero_shot \\
    --config {config_file} \\
    --start_task 0 \\
    --end_task {total_tasks}

if [ $? -ne 0 ]; then
    echo "❌ 测试失败！"
    exit 1
fi
""")
        
        f.write("""
echo "==============================================================================="
echo "✅ 快速测试完成！所有任务运行成功"
echo "==============================================================================="
""")
    
    # 设置可执行权限
    script_path.chmod(0o755)
    print(f"\n✓ 生成运行脚本: {script_path}")
    
    # 生成Windows bat脚本
    bat_path = output_path / "run_quick_test.bat"
    config_file_win = "scripts\\configs\\quick_test\\quick_test_config.json"
    
    with open(bat_path, 'w', encoding='utf-8') as f:
        f.write(f"""@echo off
REM ===============================================================================
REM 快速测试脚本 (Windows)
REM 用途：测试项目是否可以正常运行
REM 设置：epoch=2, batch_size=4, 策略=NONE
REM 任务：{total_tasks}个任务 (4个text-only + 4个multimodal)
REM ===============================================================================

setlocal enabledelayedexpansion

echo ===============================================================================
echo 快速测试 - 开始
echo ===============================================================================

cd /d "%~dp0..\..\..\"
echo 当前目录: %CD%
echo 配置文件: {config_file_win}
echo 任务数: {total_tasks}
echo.

REM 运行所有任务
echo ===============================================================================
echo 运行 {total_tasks} 个任务...
echo ===============================================================================
python -m scripts.train_with_zero_shot --config {config_file_win} --start_task 0 --end_task {total_tasks}

if errorlevel 1 (
    echo ❌ 测试失败！
    exit /b 1
)

echo ===============================================================================
echo ✅ 快速测试完成！所有任务运行成功
echo ===============================================================================
pause
""")
    
    print(f"✓ 生成Windows脚本: {bat_path}")


def main():
    parser = argparse.ArgumentParser(description="生成快速测试配置")
    parser.add_argument("--output_dir", type=str, default="scripts/configs/quick_test",
                       help="输出目录")
    
    args = parser.parse_args()
    
    print("="*80)
    print("生成快速测试配置")
    print("="*80)
    print(f"输出目录: {args.output_dir}")
    print("")
    
    # 生成配置
    config = create_quick_test_config()
    print(f"✓ 生成配置包含 {config['total_tasks']} 个任务")
    print("")
    
    # 保存配置
    save_configs(config, args.output_dir)
    
    print("")
    print("="*80)
    print("✅ 完成！")
    print("="*80)
    print(f"\n使用方法:")
    print(f"  Linux/Mac:   bash {args.output_dir}/run_quick_test.sh")
    print(f"  Windows:     {args.output_dir}\\run_quick_test.bat")
    print(f"\n或直接运行:")
    print(f"  python -m scripts.train_with_zero_shot --config {args.output_dir}/quick_test_config.json")
    print("")


if __name__ == "__main__":
    main()

