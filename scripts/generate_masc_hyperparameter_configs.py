#!/usr/bin/env python3
"""
生成MASC超参数搜索配置文件

针对任务序列: masc(text_only) -> masc(multimodal)
测试策略: none, replay, moe, deqa
搜索最佳超参数组合: lr, step_size, gamma
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple
import sys
import os

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_task_config import TaskConfigGenerator


class HyperparameterSearchGenerator:
    """超参数搜索配置生成器"""
    
    def __init__(self):
        self.base_generator = TaskConfigGenerator()
        
        # 定义超参数搜索空间
        self.hyperparameter_grid = {
            # 学习率候选值
            "lr": [5e-5, 1e-5, 5e-6],
            # step_size候选值 (学习率衰减步长)
            "step_size": [5, 10, 15],
            # gamma候选值 (学习率衰减率)
            "gamma": [0.3, 0.5, 0.7]
        }
        
        # 策略列表
        self.strategies = ["none", "replay", "moe", "deqa"]
        
    def get_hyperparameter_combinations(self) -> List[Tuple[float, int, float]]:
        """
        生成合理的超参数组合
        不是所有组合都有意义，我们选择一些有代表性的
        """
        combinations = []
        
        # 策略1: 固定gamma，变化lr和step_size
        for lr in self.hyperparameter_grid["lr"]:
            for step_size in self.hyperparameter_grid["step_size"]:
                combinations.append((lr, step_size, 0.5))  # gamma固定为0.5
        
        # 策略2: 固定lr和step_size，变化gamma
        for gamma in self.hyperparameter_grid["gamma"]:
            if gamma != 0.5:  # 避免重复
                combinations.append((1e-5, 10, gamma))
        
        # 策略3: 一些特殊的组合（基于经验）
        special_combinations = [
            (5e-5, 5, 0.7),   # 高学习率 + 快速衰减
            (5e-6, 15, 0.3),  # 低学习率 + 慢速衰减
            (1e-5, 10, 0.5),  # 中等参数（这是默认值）
        ]
        
        for combo in special_combinations:
            if combo not in combinations:
                combinations.append(combo)
        
        return combinations
    
    def generate_single_config(self, 
                              env: str,
                              dataset: str,
                              strategy: str,
                              lr: float,
                              step_size: int,
                              gamma: float,
                              seq_suffix: str = "") -> dict:
        """
        生成单个配置文件
        
        任务序列: masc(text_only) -> masc(multimodal)
        """
        # 任务序列：两个masc任务，第一个text_only，第二个multimodal
        task_sequence = ["masc", "masc"]
        mode_sequence = ["text_only", "multimodal"]
        
        # 使用基础生成器生成配置
        config = self.base_generator.generate_task_sequence_config(
            env=env,
            dataset=dataset,
            task_sequence=task_sequence,
            mode_sequence=mode_sequence,
            strategy=strategy,
            use_label_embedding=False,
            seq_suffix=seq_suffix,
            lr=lr,
            step_size=step_size,
            gamma=gamma,
            epochs=20,      # 固定20个epoch
            patience=999    # 禁用早停，确保完整训练以公平比较超参数
        )
        
        # 添加超参数信息到配置中
        config["hyperparameters"] = {
            "lr": lr,
            "step_size": step_size,
            "gamma": gamma
        }
        
        return config
    
    def generate_all_configs(self, 
                            env: str = "server",
                            dataset: str = "twitter2015",
                            output_dir: str = "scripts/configs/hyperparam_search"):
        """
        生成所有配置文件
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取超参数组合
        hyperparams = self.get_hyperparameter_combinations()
        
        print(f"生成配置文件到: {output_path}")
        print(f"策略数量: {len(self.strategies)}")
        print(f"超参数组合数量: {len(hyperparams)}")
        print(f"总配置文件数: {len(self.strategies) * len(hyperparams)}")
        print()
        
        configs_generated = []
        
        for strategy in self.strategies:
            for i, (lr, step_size, gamma) in enumerate(hyperparams):
                # 生成配置文件名
                lr_str = f"{lr:.0e}".replace("-", "").replace("+", "")  # 5e-05 -> 5e05
                config_name = f"{env}_{dataset}_{strategy}_lr{lr_str}_ss{step_size}_g{gamma:.1f}.json"
                config_file = output_path / config_name
                
                # 生成序列后缀
                seq_suffix = f"hp{i+1}"
                
                # 生成配置
                config = self.generate_single_config(
                    env=env,
                    dataset=dataset,
                    strategy=strategy,
                    lr=lr,
                    step_size=step_size,
                    gamma=gamma,
                    seq_suffix=seq_suffix
                )
                
                # 保存配置
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                # 保存时使用Unix风格路径（跨平台兼容）
                configs_generated.append({
                    "file": config_file.as_posix(),
                    "strategy": strategy,
                    "lr": lr,
                    "step_size": step_size,
                    "gamma": gamma
                })
                
                print(f"✓ {config_name}")
        
        # 生成索引文件
        index_file = output_path / "config_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_configs": len(configs_generated),
                "strategies": self.strategies,
                "hyperparameter_grid": self.hyperparameter_grid,
                "configs": configs_generated
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n索引文件已生成: {index_file}")
        print(f"\n总共生成 {len(configs_generated)} 个配置文件")
        
        return configs_generated


def validate_config(config_file: str) -> bool:
    """验证配置文件是否正确"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查必要的字段
        required_fields = ["tasks", "global_params", "strategy", "env", "dataset"]
        for field in required_fields:
            if field not in config:
                print(f"❌ 缺少必要字段: {field}")
                return False
        
        # 检查任务配置
        if len(config["tasks"]) != 2:
            print(f"❌ 任务数量错误: 期望2，实际{len(config['tasks'])}")
            return False
        
        # 检查任务名称和模式
        task1 = config["tasks"][0]
        task2 = config["tasks"][1]
        
        if task1["task_name"] != "masc" or task2["task_name"] != "masc":
            print(f"❌ 任务名称错误: {task1['task_name']}, {task2['task_name']}")
            return False
        
        if task1["mode"] != "text_only" or task2["mode"] != "multimodal":
            print(f"❌ 模式错误: {task1['mode']}, {task2['mode']}")
            return False
        
        # 检查超参数
        if "hyperparameters" in config:
            hp = config["hyperparameters"]
            print(f"  超参数: lr={hp['lr']}, step_size={hp['step_size']}, gamma={hp['gamma']}")
        
        print(f"✓ 配置文件有效")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="生成MASC超参数搜索配置")
    parser.add_argument("--env", type=str, default="server",
                       choices=["local", "server"],
                       help="环境类型")
    parser.add_argument("--dataset", type=str, default="twitter2015",
                       choices=["twitter2015", "twitter2017", "mix"],
                       help="数据集名称")
    parser.add_argument("--output_dir", type=str, 
                       default="scripts/configs/hyperparam_search",
                       help="输出目录")
    parser.add_argument("--validate", action="store_true",
                       help="生成后验证所有配置文件")
    parser.add_argument("--start_exp_id", type=int, default=1,
                       help="从第几个实验开始（用于恢复或继续实验）")
    
    args = parser.parse_args()
    
    # 生成配置
    generator = HyperparameterSearchGenerator()
    configs = generator.generate_all_configs(
        env=args.env,
        dataset=args.dataset,
        output_dir=args.output_dir
    )
    
    # 验证配置
    if args.validate:
        print("\n" + "=" * 60)
        print("验证配置文件...")
        print("=" * 60)
        
        valid_count = 0
        for config_info in configs:
            config_file = config_info["file"]
            print(f"\n检查: {Path(config_file).name}")
            if validate_config(config_file):
                valid_count += 1
        
        print(f"\n验证完成: {valid_count}/{len(configs)} 个配置文件有效")
    
    # 生成运行脚本 - 支持多GPU并行
    run_script_path = Path(args.output_dir) / "run_all_experiments.sh"
    _generate_multi_gpu_script(run_script_path, configs, args.output_dir, args.start_exp_id)
    
    # 生成后台启动脚本
    wrapper_script_path = Path(args.output_dir) / "start_experiments_detached.sh"
    _generate_wrapper_script(wrapper_script_path, run_script_path, args.output_dir)
    
    # 生成README
    readme_path = Path(args.output_dir) / "README.md"
    _generate_readme(readme_path, len(configs), args.output_dir)
    
    print(f"\n✓ 运行脚本已生成: {run_script_path}")
    print(f"✓ 后台启动脚本已生成: {wrapper_script_path}")
    print(f"✓ 使用说明已生成: {readme_path}")
    if args.start_exp_id > 1:
        print(f"\n注意：脚本配置为从实验 #{args.start_exp_id} 开始")
        print(f"  实验 #1 到 #{args.start_exp_id-1} 将被跳过")
    print(f"\n推荐使用方法（可脱离SSH）:")
    print(f"  bash {wrapper_script_path}")
    print(f"\n或者使用 tmux/screen:")
    print(f"  tmux new -s masc_hyperparam")
    print(f"  bash {run_script_path}")


def _generate_multi_gpu_script(script_path: Path, configs: list, output_dir: str, start_exp_id: int = 1):
    """生成支持多GPU并行的运行脚本"""
    
    # 使用newline='\n'强制使用Unix换行符，避免Windows的\r\n问题
    with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write("""#!/bin/bash
#===============================================================================
# MASC 超参数搜索实验脚本 - 多GPU优化版
# 
# 功能:
#   - 自动检测可用GPU数量
#   - 智能分配实验到不同GPU（充分利用资源）
#   - 支持SSH断开后继续运行 (使用nohup)
#   - 独立日志文件
#   - GPU使用监控（基于GPU使用率）
#   - 支持从指定实验ID开始
#
# GPU分配策略:
#   - 优先使用GPU使用率<10%的GPU
#   - 等待超时后进行连续检测（1分钟内每5秒检查一次）
#   - 连续检测标准：GPU使用率<20% 且 显存使用<20%
#   - 必须连续12次检查都通过才能使用GPU
#   - 避免干扰其他正在运行的程序，防止因任务启动/结束时的瞬时波动导致冲突
#
# 优化策略:
#   2张GPU: 实验自动轮流分配到GPU0和GPU1
#   多张GPU: 充分并行利用所有GPU
#
# 使用方法:
#   bash {script_name}
#
# 日志位置:
#   {log_dir}/exp_{{id}}_{{timestamp}}.log
#===============================================================================

# 设置错误时退出
set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
# 脚本在 scripts/configs/hyperparam_search/ 下，需要向上3层到达项目根目录
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 打印当前目录便于调试
echo "项目根目录: $PROJECT_ROOT"
echo "当前工作目录: $(pwd)"

# 创建日志目录
LOG_DIR="{log_dir}"
mkdir -p "$LOG_DIR"

# 获取时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 颜色定义
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
CYAN='\\033[0;36m'
MAGENTA='\\033[0;35m'
NC='\\033[0m' # No Color

#===============================================================================
# 辅助函数
#===============================================================================

# 打印分隔线
print_separator() {{
    echo -e "${{BLUE}}===============================================================================${{NC}}"
}}

# 打印信息
print_info() {{
    echo -e "${{GREEN}}[INFO]${{NC}} $1"
}}

# 打印警告
print_warning() {{
    echo -e "${{YELLOW}}[WARNING]${{NC}} $1"
}}

# 打印错误
print_error() {{
    echo -e "${{RED}}[ERROR]${{NC}} $1"
}}

# 打印GPU信息
print_gpu_info() {{
    echo -e "${{CYAN}}[GPU]${{NC}} $1"
}}

# 检测可用GPU数量
detect_gpus() {{
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        echo $gpu_count
    else
        print_warning "nvidia-smi未找到，假设有1个GPU"
        echo 1
    fi
}}

# 获取空闲GPU（GPU使用率最低的GPU）
get_free_gpu() {{
    if command -v nvidia-smi &> /dev/null; then
        # 获取所有GPU的使用率情况，找到使用率最低的
        # CSV格式用逗号分隔，需要正确处理
        local free_gpu=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | \\
                        sed 's/,/ /g' | \\
                        awk '{{print $1, $2}}' | \\
                        sort -k2 -n | \\
                        head -1 | \\
                        awk '{{print $1}}')
        echo "$free_gpu"
    else
        echo "0"
    fi
}}

# 等待空闲GPU
wait_for_free_gpu() {{
    local max_wait=${{1:-3600}}  # 默认最多等待1小时
    local wait_time=0
    local check_interval=30  # 每30秒检查一次
    
    print_info "开始寻找空闲GPU..." >&2
    
    while [ $wait_time -lt $max_wait ]; do
        local free_gpu=$(get_free_gpu)
        
        # 验证GPU ID是否有效（应该是数字）
        if ! [[ "$free_gpu" =~ ^[0-9]+$ ]]; then
            # GPU ID无效，重新获取
            free_gpu=$(get_free_gpu)
            if ! [[ "$free_gpu" =~ ^[0-9]+$ ]]; then
                free_gpu=0
            fi
        fi
        
        # 获取GPU使用率（添加错误处理）
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $free_gpu 2>/dev/null | sed 's/,//g' | awk '{{print $1}}')
        
        # 如果无法获取GPU使用率，默认认为可用
        if [ -z "$gpu_util" ]; then
            echo "$free_gpu"
            return 0
        fi
        
        # 如果GPU使用率 < 10%，进行连续检测确认
        if [ "$gpu_util" -lt 10 ] 2>/dev/null; then
            print_info "发现GPU${{free_gpu}}使用率较低(${{gpu_util}}%)，开始连续检测验证..." >&2
            
            local continuous_check_duration=60
            local continuous_check_interval=5
            local checks_needed=$((continuous_check_duration / continuous_check_interval))
            local checks_passed=0
            
            for ((i=1; i<=checks_needed; i++)); do
                local current_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $free_gpu 2>/dev/null | sed 's/,//g' | awk '{{print $1}}')
                local mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $free_gpu 2>/dev/null | sed 's/,//g' | awk '{{print $1}}')
                local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $free_gpu 2>/dev/null | sed 's/,//g' | awk '{{print $1}}')
                
                local mem_percent=0
                if [ -n "$mem_total" ] && [ "$mem_total" -gt 0 ] 2>/dev/null; then
                    mem_percent=$((mem_used * 100 / mem_total))
                fi
                
                # 连续检测标准：使用率 < 10% 且 显存 < 10%
                if [ "$current_util" -lt 10 ] 2>/dev/null && [ "$mem_percent" -lt 10 ] 2>/dev/null; then
                    checks_passed=$((checks_passed + 1))
                    print_info "  检查 $i/${{checks_needed}}: ✓ 通过 (使用率: ${{current_util}}%, 显存: ${{mem_percent}}%)" >&2
                else
                    print_warning "  检查 $i/${{checks_needed}}: ✗ 未通过 (使用率: ${{current_util}}%, 显存: ${{mem_percent}}%)，放弃此GPU" >&2
                    break
                fi
                
                if [ $i -lt $checks_needed ]; then
                    sleep $continuous_check_interval
                fi
            done
            
            # 如果连续检测通过，使用该GPU
            if [ $checks_passed -eq $checks_needed ]; then
                print_info "GPU${{free_gpu}}连续检测通过（使用率<10%且显存<10%），可以使用" >&2
                echo "$free_gpu"
                return 0
            else
                print_warning "GPU${{free_gpu}}连续检测未通过，继续等待..." >&2
            fi
        fi
        
        # 输出警告到stderr，不影响返回值
        print_warning "所有GPU都在使用中 (GPU$free_gpu: ${{gpu_util}}%)，等待 ${{check_interval}} 秒后重试..." >&2
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    # 超时后进入更宽松的连续检测模式
    print_warning "等待超时，进入宽松检测模式（1分钟内使用率<20%且显存<20%）..." >&2
    
    while true; do
        local fallback_gpu=$(get_free_gpu)
        local continuous_check_duration=60  # 连续检测1分钟
        local continuous_check_interval=5   # 每5秒检查一次
        local checks_needed=$((continuous_check_duration / continuous_check_interval))
        local checks_passed=0
        
        print_info "正在对GPU${{fallback_gpu}}进行连续检测（需要连续${{checks_needed}}次通过）..." >&2
        
        # 连续检测
        for ((i=1; i<=checks_needed; i++)); do
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $fallback_gpu 2>/dev/null | sed 's/,//g' | awk '{{print $1}}')
            local mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $fallback_gpu 2>/dev/null | sed 's/,//g' | awk '{{print $1}}')
            local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $fallback_gpu 2>/dev/null | sed 's/,//g' | awk '{{print $1}}')
            
            # 计算显存使用率
            local mem_percent=0
            if [ -n "$mem_total" ] && [ "$mem_total" -gt 0 ] 2>/dev/null; then
                mem_percent=$((mem_used * 100 / mem_total))
            fi
            
            # 宽松检测标准：GPU使用率<20% 且 显存使用<20%
            if [ "$gpu_util" -lt 20 ] 2>/dev/null && [ "$mem_percent" -lt 20 ] 2>/dev/null; then
                checks_passed=$((checks_passed + 1))
                print_info "  检查 $i/${{checks_needed}}: ✓ 通过 (使用率: ${{gpu_util}}%, 显存: ${{mem_percent}}%)" >&2
            else
                print_warning "  检查 $i/${{checks_needed}}: ✗ 未通过 (使用率: ${{gpu_util}}%, 显存: ${{mem_percent}}%)，重新开始检测..." >&2
                break
            fi
            
            # 如果还需要继续检查，等待一段时间
            if [ $i -lt $checks_needed ]; then
                sleep $continuous_check_interval
            fi
        done
        
        # 如果所有检查都通过，返回该GPU
        if [ $checks_passed -eq $checks_needed ]; then
            print_info "GPU${{fallback_gpu}}连续检测通过（使用率<20%且显存<20%），可以使用" >&2
            echo "$fallback_gpu"
            return 0
        fi
        
        # 否则，等待一段时间后重新开始整个连续检测流程
        print_warning "GPU${{fallback_gpu}}连续检测未通过，等待30秒后重新检测..." >&2
        sleep 30
    done
}}

# 运行单个实验
run_experiment() {{
    local exp_id=$1
    local config_file=$2
    local strategy=$3
    local lr=$4
    local step_size=$5
    local gamma=$6
    local gpu_id=$7
    
    local log_file="${{LOG_DIR}}/exp_${{exp_id}}_${{strategy}}_${{TIMESTAMP}}.log"
    local pid_file="${{LOG_DIR}}/exp_${{exp_id}}.pid"
    
    print_separator
    print_info "启动实验 #${{exp_id}}"
    print_info "  策略: ${{strategy}}"
    print_info "  超参数: lr=${{lr}}, step_size=${{step_size}}, gamma=${{gamma}}"
    print_gpu_info "  使用GPU: ${{gpu_id}}"
    print_info "  日志文件: ${{log_file}}"
    
    # 设置CUDA_VISIBLE_DEVICES，只使用指定的GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # 使用nohup在后台运行，添加任务范围参数
    nohup python -m scripts.train_with_zero_shot --config "${{config_file}}" --start_task 0 --end_task 2 > "${{log_file}}" 2>&1 &
    
    local pid=$!
    echo $pid > "${{pid_file}}"
    
    print_info "  进程ID: ${{pid}}"
    print_info "  PID文件: ${{pid_file}}"
    
    # 取消CUDA_VISIBLE_DEVICES的导出
    unset CUDA_VISIBLE_DEVICES
    
    # 等待一小段时间，确保进程启动
    sleep 5
    
    # 检查进程是否还在运行
    if ps -p $pid > /dev/null; then
        print_info "  实验 #${{exp_id}} 已成功启动"
    else
        print_error "  实验 #${{exp_id}} 启动失败"
        return 1
    fi
    
    return 0
}}

# 等待所有后台任务完成
wait_all_experiments() {{
    print_separator
    print_info "等待所有实验完成..."
    
    local all_done=false
    while [ "$all_done" = false ]; do
        all_done=true
        
        for pid_file in "${{LOG_DIR}}"/*.pid; do
            if [ -f "$pid_file" ]; then
                local pid=$(cat "$pid_file")
                if ps -p $pid > /dev/null; then
                    all_done=false
                    break
                fi
            fi
        done
        
        if [ "$all_done" = false ]; then
            sleep 60  # 每分钟检查一次
        fi
    done
    
    print_info "所有实验已完成"
}}

#===============================================================================
# 主程序
#===============================================================================

print_separator
print_info "MASC 超参数搜索实验启动"
print_separator

# 检测GPU数量
GPU_COUNT=$(detect_gpus)
print_gpu_info "检测到 ${{GPU_COUNT}} 个GPU"

# 显示GPU信息
if command -v nvidia-smi &> /dev/null; then
    print_separator
    nvidia-smi
    print_separator
fi

# 开始实验
print_info "开始运行实验..."
print_separator

""".format(
            script_name=script_path.name,
            log_dir=f"{output_dir}/logs",
        ))
        
        # 生成每个实验的运行命令
        exp_id = start_exp_id
        for i, config_info in enumerate(configs):
            # 如果当前实验索引小于起始索引，跳过
            if i + 1 < start_exp_id:
                continue
            
            # 转换为Unix风格路径（正斜杠）
            config_file = Path(config_info["file"]).as_posix()
            strategy = config_info["strategy"]
            lr = config_info["lr"]
            step_size = config_info["step_size"]
            gamma = config_info["gamma"]
            
            f.write(f"""
# 实验 {exp_id}: {strategy} - lr={lr}, step_size={step_size}, gamma={gamma}
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment {exp_id} "{config_file}" "{strategy}" {lr} {step_size} {gamma} $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 {exp_id}"
fi
""")
            exp_id += 1
        
        # 添加结束部分
        f.write("""
# 等待所有实验完成
wait_all_experiments

print_separator
print_info "所有实验已完成！"
print_separator

# 清理PID文件
rm -f "${LOG_DIR}"/*.pid

# 显示结果摘要
print_info "实验日志位置: ${LOG_DIR}"
print_info "查看实验结果:"
echo ""
for log_file in "${LOG_DIR}"/exp_*_${TIMESTAMP}.log; do
    if [ -f "$log_file" ]; then
        echo "  - $log_file"
    fi
done

print_separator
""")


def _generate_wrapper_script(wrapper_path: Path, main_script_path: Path, output_dir: str):
    """生成可以完全脱离SSH运行的包装脚本"""
    
    log_dir = f"{output_dir}/logs"
    master_log = f"{log_dir}/master_$(date +%Y%m%d_%H%M%S).log"
    
    # 使用newline='\n'强制使用Unix换行符
    with open(wrapper_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f"""#!/bin/bash
#===============================================================================
# MASC 超参数搜索 - 后台启动脚本
# 
# 此脚本可以完全脱离SSH运行，即使SSH断开连接也会继续执行所有实验
#
# 使用方法:
#   bash {wrapper_path.name}
#
# 检查运行状态:
#   tail -f {log_dir}/master_*.log
#   ps aux | grep train_with_zero_shot
#
# 停止所有实验:
#   bash {output_dir}/stop_all_experiments.sh
#===============================================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/{main_script_path.name}"
LOG_DIR="{log_dir}"
MASTER_LOG="{master_log}"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 打印路径信息便于调试
echo "脚本目录: $SCRIPT_DIR"
echo "主脚本: $MAIN_SCRIPT"

# 颜色定义
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
CYAN='\\033[0;36m'
NC='\\033[0m'

echo -e "${{GREEN}}========================================================================${{NC}}"
echo -e "${{GREEN}}MASC 超参数搜索实验 - 后台启动${{NC}}"
echo -e "${{GREEN}}========================================================================${{NC}}"
echo ""

# 检查主脚本是否存在
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo -e "${{YELLOW}}[ERROR]${{NC}} 主脚本不存在: $MAIN_SCRIPT"
    exit 1
fi

# 使用nohup启动主脚本，完全脱离当前shell
echo -e "${{CYAN}}[INFO]${{NC}} 启动主实验脚本..."
echo -e "${{CYAN}}[INFO]${{NC}} 主日志文件: $MASTER_LOG"
echo ""

nohup bash "$MAIN_SCRIPT" > "$MASTER_LOG" 2>&1 &
MASTER_PID=$!

# 保存主进程PID
echo $MASTER_PID > "${{LOG_DIR}}/master.pid"

echo -e "${{GREEN}}✓ 主脚本已在后台启动${{NC}}"
echo -e "${{GREEN}}✓ 主进程PID: $MASTER_PID${{NC}}"
echo -e "${{GREEN}}✓ PID文件: ${{LOG_DIR}}/master.pid${{NC}}"
echo ""

# 等待一下确保进程启动
sleep 3

# 检查进程是否还在运行
if ps -p $MASTER_PID > /dev/null; then
    echo -e "${{GREEN}}✓ 实验已成功启动并在后台运行${{NC}}"
    echo ""
    echo -e "${{CYAN}}========================================================================${{NC}}"
    echo -e "${{CYAN}}监控命令:${{NC}}"
    echo -e "${{CYAN}}========================================================================${{NC}}"
    echo -e "  查看主日志:   tail -f $MASTER_LOG"
    echo -e "  查看实验日志: ls -lth $LOG_DIR/exp_*.log"
    echo -e "  检查进程:     ps aux | grep train_with_zero_shot"
    echo -e "  查看GPU使用:  watch -n 1 nvidia-smi"
    echo ""
    echo -e "${{CYAN}}停止命令:${{NC}}"
    echo -e "  停止所有实验: bash $SCRIPT_DIR/stop_all_experiments.sh"
    echo ""
    echo -e "${{GREEN}}========================================================================${{NC}}"
    echo -e "${{GREEN}}现在可以安全地断开SSH连接，实验将继续运行${{NC}}"
    echo -e "${{GREEN}}========================================================================${{NC}}"
else
    echo -e "${{YELLOW}}[ERROR]${{NC}} 主脚本启动失败，请检查日志: $MASTER_LOG"
    exit 1
fi
""")
    
    # 设置可执行权限
    wrapper_path.chmod(0o755)
    
    # 生成停止脚本
    stop_script_path = wrapper_path.parent / "stop_all_experiments.sh"
    # 使用newline='\n'强制使用Unix换行符
    with open(stop_script_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f"""#!/bin/bash
#===============================================================================
# 停止所有MASC超参数搜索实验
#===============================================================================

LOG_DIR="{log_dir}"
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m'

echo -e "${{RED}}========================================================================${{NC}}"
echo -e "${{RED}}停止所有实验${{NC}}"
echo -e "${{RED}}========================================================================${{NC}}"
echo ""

# 停止主进程
if [ -f "${{LOG_DIR}}/master.pid" ]; then
    MASTER_PID=$(cat "${{LOG_DIR}}/master.pid")
    if ps -p $MASTER_PID > /dev/null; then
        echo -e "${{YELLOW}}停止主进程 (PID: $MASTER_PID)...${{NC}}"
        kill $MASTER_PID
    fi
    rm -f "${{LOG_DIR}}/master.pid"
fi

# 停止所有实验进程
for pid_file in "${{LOG_DIR}}"/*.pid; do
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null; then
            echo -e "${{YELLOW}}停止实验进程 (PID: $PID)...${{NC}}"
            kill $PID
        fi
        rm -f "$pid_file"
    fi
done

# 确保所有Python训练进程都被停止
echo ""
echo -e "${{YELLOW}}检查是否还有训练进程在运行...${{NC}}"
TRAIN_PIDS=$(ps aux | grep "train_with_zero_shot" | grep -v grep | awk '{{print $2}}')

if [ -n "$TRAIN_PIDS" ]; then
    echo -e "${{YELLOW}}发现残留进程，强制终止...${{NC}}"
    echo "$TRAIN_PIDS" | xargs kill -9 2>/dev/null || true
fi

echo ""
echo -e "${{GREEN}}✓ 所有实验已停止${{NC}}"
echo ""
""")
    
    stop_script_path.chmod(0o755)


def _generate_readme(readme_path: Path, total_configs: int, output_dir: str):
    """生成使用说明README"""
    
    # 使用newline='\n'强制使用Unix换行符
    with open(readme_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f"""# MASC 超参数搜索实验

## 概述

本目录包含 **{total_configs}** 个超参数搜索实验配置，用于MASC任务序列的最优超参数搜索。

**任务序列**: masc(text_only) -> masc(multimodal)  
**测试策略**: none, replay, moe, deqa  
**超参数**: lr, step_size, gamma

## 文件说明

- `config_index.json` - 所有配置文件的索引
- `*.json` - 各个实验的配置文件
- `run_all_experiments.sh` - 主实验运行脚本（需要SSH保持连接）
- `start_experiments_detached.sh` - **推荐**：后台启动脚本（可脱离SSH）
- `stop_all_experiments.sh` - 停止所有实验
- `logs/` - 实验日志目录

## 使用方法

### ⚠️ 重要提示

脚本会自动切换到项目根目录运行，确保：
1. 脚本从正确的位置执行
2. 能够正确导入 `scripts.train_with_zero_shot` 模块
3. 日志输出会显示当前工作目录，请检查是否正确

### 生成配置时的选项

如果需要从特定实验开始（例如已完成前2个实验），在生成配置时使用：

```bash
# 从第3个实验开始生成脚本
python -m scripts.generate_masc_hyperparameter_configs --start_exp_id 3

# 这样生成的脚本会从exp_3开始编号，避免覆盖已有结果
```

### 方法1：后台运行（推荐，可脱离SSH）

```bash
# 启动所有实验（可以断开SSH）
bash {output_dir}/start_experiments_detached.sh

# 断开SSH连接也没问题，实验会继续运行

# 检查主日志，确认项目根目录是否正确
tail -f {output_dir}/logs/master_*.log | head -20
```

### 方法2：使用 tmux（推荐）

```bash
# 创建tmux会话
tmux new -s masc_hyperparam

# 运行实验
bash {output_dir}/run_all_experiments.sh

# 断开会话（实验继续运行）：Ctrl+B 然后按 D
# 重新连接：tmux attach -t masc_hyperparam
```

### 方法3：使用 screen

```bash
# 创建screen会话
screen -S masc_hyperparam

# 运行实验
bash {output_dir}/run_all_experiments.sh

# 断开会话：Ctrl+A 然后按 D
# 重新连接：screen -r masc_hyperparam
```

## 监控实验

### 查看主日志
```bash
tail -f {output_dir}/logs/master_*.log
```

### 查看单个实验日志
```bash
ls -lth {output_dir}/logs/exp_*.log
tail -f {output_dir}/logs/exp_1_none_*.log
```

### 查看所有运行中的进程
```bash
ps aux | grep train_with_zero_shot
```

### 监控GPU使用
```bash
watch -n 1 nvidia-smi
```

### 查看实验进度
```bash
# 查看有多少实验已完成
grep "completed successfully" {output_dir}/logs/*.log | wc -l

# 查看有多少实验正在运行
ps aux | grep train_with_zero_shot | grep -v grep | wc -l
```

## 停止实验

### 停止所有实验
```bash
bash {output_dir}/stop_all_experiments.sh
```

### 停止单个实验
```bash
# 查找PID
ps aux | grep train_with_zero_shot

# 停止特定进程
kill <PID>
```

## GPU分配策略

脚本会自动检测可用GPU并智能分配：

- **2张GPU**: 实验会轮流分配到GPU0和GPU1，充分利用两张卡
- **多张GPU**: 自动并行利用所有可用GPU
- **单张GPU**: 实验会串行执行，前一个完成后启动下一个

### GPU空闲判断标准

- **快速检测阶段**: GPU使用率 < 10% → 立即使用
- **等待超时后的严格检测**:
  - 进入连续检测模式（1分钟，每5秒检查一次，共12次）
  - 连续检测标准：**GPU使用率 < 20% 且 显存使用 < 20%**
  - 必须**连续12次检查都通过**才能使用GPU
  - 如果任何一次检查未通过，重新开始整个检测流程
  - **目的**: 避免因任务启动/结束时的瞬时波动导致GPU冲突

## 实验配置

总共 **{total_configs}** 个配置，包括：

- 4种策略 (none, replay, moe, deqa)
- 多组超参数组合 (lr, step_size, gamma)
- 每个实验固定20个epoch
- **早停已禁用** (patience=999)，确保所有实验完整训练以公平比较

详细配置请查看 `config_index.json`

## 结果分析

实验完成后，结果会保存在：

- 模型文件: `checkpoints/twitter2015_<strategy>_<mode>_<seq>.pt`
- 训练信息: `checkpoints/train_info_twitter2015_<strategy>_<mode>_<seq>.json`
- 准确率热力图: `checkpoints/acc_matrix/`

## 故障排查

### 问题：No module named scripts.train_with_zero_shot
```bash
# 这通常是因为工作目录不正确
# 检查主日志中的"项目根目录"和"当前工作目录"
tail -n 50 {output_dir}/logs/master_*.log | grep "目录"

# 应该看到类似输出：
# 项目根目录: /path/to/MCM
# 当前工作目录: /path/to/MCM

# 如果路径不对，手动运行脚本时请从项目根目录执行
```

### 问题：进程启动失败
```bash
# 检查主日志
tail -n 50 {output_dir}/logs/master_*.log

# 检查实验日志
tail -n 50 {output_dir}/logs/exp_1_*.log
```

### 问题：GPU显存不足
```bash
# 查看GPU使用情况
nvidia-smi

# 可能需要减少并行实验数量，编辑 run_all_experiments.sh
# 在 wait_for_free_gpu 函数中调整显存阈值
```

### 问题：SSH断开后进程停止
```bash
# 确保使用了后台启动脚本
bash {output_dir}/start_experiments_detached.sh

# 或者使用tmux/screen
```

## 注意事项

1. **确保有足够的磁盘空间**：每个实验会生成模型文件和日志
2. **监控GPU温度**：长时间运行可能导致GPU过热
3. **定期检查日志**：及时发现和处理错误
4. **实验命名**：配置文件名已经包含了超参数信息，便于识别

## 联系方式

如有问题，请查看主项目README或联系开发者。
""")


if __name__ == "__main__":
    main()

