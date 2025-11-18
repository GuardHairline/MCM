#!/usr/bin/env python3
"""
任务配置文件生成器

生成包含所有任务信息的JSON配置文件，用于持续学习训练。
这样可以在训练第i个任务时，预先知道后续任务的信息，实现0样本检测。
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def _generate_run_scripts(output_dir: str, configs: List[Dict]):
    """生成运行脚本"""
    output_path = Path(output_dir)
    
    # 1. 生成主运行脚本
    run_script_path = output_path / "run_all_experiments.sh"
    _generate_main_script(run_script_path, configs, output_dir)
    
    # 2. 生成后台启动脚本
    start_script_path = output_path / "start_all_experiments.sh"
    _generate_start_script(start_script_path, run_script_path, output_dir)
    
    # 3. 生成停止脚本
    stop_script_path = output_path / "stop_all_experiments.sh"
    _generate_stop_script(stop_script_path, output_dir)
    
    # 设置可执行权限
    run_script_path.chmod(0o755)
    start_script_path.chmod(0o755)
    stop_script_path.chmod(0o755)


def _generate_main_script(script_path: Path, configs: List[Dict], output_dir: str):
    """生成主运行脚本 - 复用GPU检测逻辑"""
    
    with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write("""#!/bin/bash
#===============================================================================
# 全任务序列实验脚本 - 多GPU优化版
# 
# 运行顺序:
#   for seq in [seq1, seq2]:
#       for dataset in [twitter2015, twitter2017, mix]:
#           for strategy in [deqa, none, moe, replay, ...]:
#               运行8个任务
#
# GPU分配策略:
#   - 优先使用GPU使用率<10%的GPU
#   - 等待超时后进行连续检测（1分钟内每5秒检查一次）
#   - 连续检测标准：GPU使用率<20% 且 显存使用<20%
#   - 必须连续12次检查都通过才能使用GPU
#===============================================================================

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

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
NC='\\033[0m'

#===============================================================================
# 辅助函数（GPU检测逻辑）
#===============================================================================

print_separator() {{
    echo -e "${{BLUE}}===============================================================================${{NC}}"
}}

print_info() {{
    echo -e "${{GREEN}}[INFO]${{NC}} $1"
}}

print_warning() {{
    echo -e "${{YELLOW}}[WARNING]${{NC}} $1"
}}

print_error() {{
    echo -e "${{RED}}[ERROR]${{NC}} $1"
}}

print_gpu_info() {{
    echo -e "${{CYAN}}[GPU]${{NC}} $1"
}}

detect_gpus() {{
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        echo $gpu_count
    else
        print_warning "nvidia-smi未找到，假设有1个GPU"
        echo 1
    fi
}}

get_free_gpu() {{
    if command -v nvidia-smi &> /dev/null; then
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

wait_for_free_gpu() {{
    local max_wait=${{1:-3600}}
    local wait_time=0
    local check_interval=30
    
    print_info "开始寻找空闲GPU..." >&2
    
    while [ $wait_time -lt $max_wait ]; do
        local free_gpu=$(get_free_gpu)
        
        if ! [[ "$free_gpu" =~ ^[0-9]+$ ]]; then
            free_gpu=$(get_free_gpu)
            if ! [[ "$free_gpu" =~ ^[0-9]+$ ]]; then
                free_gpu=0
            fi
        fi
        
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $free_gpu 2>/dev/null | sed 's/,//g' | awk '{{print $1}}')
        
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
        
        print_warning "所有GPU都在使用中 (GPU$free_gpu: ${{gpu_util}}%)，等待 ${{check_interval}} 秒后重试..." >&2
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    # 超时后进入更宽松的连续检测模式
    print_warning "等待超时，进入宽松检测模式（1分钟内使用率<20%且显存<20%）..." >&2
    
    while true; do
        local fallback_gpu=$(get_free_gpu)
        local continuous_check_duration=60
        local continuous_check_interval=5
        local checks_needed=$((continuous_check_duration / continuous_check_interval))
        local checks_passed=0
        
        print_info "正在对GPU${{fallback_gpu}}进行连续检测（需要连续${{checks_needed}}次通过）..." >&2
        
        for ((i=1; i<=checks_needed; i++)); do
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $fallback_gpu 2>/dev/null | sed 's/,//g' | awk '{{print $1}}')
            local mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $fallback_gpu 2>/dev/null | sed 's/,//g' | awk '{{print $1}}')
            local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $fallback_gpu 2>/dev/null | sed 's/,//g' | awk '{{print $1}}')
            
            local mem_percent=0
            if [ -n "$mem_total" ] && [ "$mem_total" -gt 0 ] 2>/dev/null; then
                mem_percent=$((mem_used * 100 / mem_total))
            fi
            
            # 宽松检测标准：使用率 < 20% 且 显存 < 20%
            if [ "$gpu_util" -lt 20 ] 2>/dev/null && [ "$mem_percent" -lt 20 ] 2>/dev/null; then
                checks_passed=$((checks_passed + 1))
                print_info "  检查 $i/${{checks_needed}}: ✓ 通过 (使用率: ${{gpu_util}}%, 显存: ${{mem_percent}}%)" >&2
            else
                print_warning "  检查 $i/${{checks_needed}}: ✗ 未通过 (使用率: ${{gpu_util}}%, 显存: ${{mem_percent}}%)，重新开始检测..." >&2
                break
            fi
            
            if [ $i -lt $checks_needed ]; then
                sleep $continuous_check_interval
            fi
        done
        
        if [ $checks_passed -eq $checks_needed ]; then
            print_info "GPU${{fallback_gpu}}连续检测通过（使用率<20%且显存<20%），可以使用" >&2
            echo "$fallback_gpu"
            return 0
        fi
        
        print_warning "GPU${{fallback_gpu}}连续检测未通过，等待30秒后重新检测..." >&2
        sleep 30
    done
}}

run_experiment() {{
    local exp_id=$1
    local config_file=$2
    local seq=$3
    local dataset=$4
    local strategy=$5
    local gpu_id=$6
    
    local log_file="${{LOG_DIR}}/exp_${{exp_id}}_${{seq}}_${{dataset}}_${{strategy}}_${{TIMESTAMP}}.log"
    local pid_file="${{LOG_DIR}}/exp_${{exp_id}}.pid"
    
    print_separator
    print_info "启动实验 #${{exp_id}}"
    print_info "  序列: ${{seq}}"
    print_info "  数据集: ${{dataset}}"
    print_info "  策略: ${{strategy}}"
    print_gpu_info "  使用GPU: ${{gpu_id}}"
    print_info "  日志文件: ${{log_file}}"
    
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # 运行8个任务（0-7）
    nohup python -m scripts.train_with_zero_shot --config "${{config_file}}" --start_task 0 --end_task 8 > "${{log_file}}" 2>&1 &
    
    local pid=$!
    echo $pid > "${{pid_file}}"
    
    print_info "  进程ID: ${{pid}}"
    print_info "  PID文件: ${{pid_file}}"
    
    unset CUDA_VISIBLE_DEVICES
    
    sleep 5
    
    if ps -p $pid > /dev/null; then
        print_info "  实验 #${{exp_id}} 已成功启动"
    else
        print_error "  实验 #${{exp_id}} 启动失败"
        return 1
    fi
    
    return 0
}}

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
            sleep 60
        fi
    done
    
    print_info "所有实验已完成"
}}

#===============================================================================
# 主程序
#===============================================================================

print_separator
print_info "全任务序列实验启动"
print_separator

GPU_COUNT=$(detect_gpus)
print_gpu_info "检测到 ${{GPU_COUNT}} 个GPU"

if command -v nvidia-smi &> /dev/null; then
    print_separator
    nvidia-smi
    print_separator
fi

print_info "开始运行实验..."
print_separator

""".format(log_dir=f"{output_dir}/logs"))
        
        # 生成每个实验的运行命令
        exp_id = 1
        for config_info in configs:
            config_file = Path(config_info["file"]).as_posix()
            seq = config_info["seq"]
            dataset = config_info["dataset"]
            strategy = config_info["strategy"]
            
            f.write(f"""
# 实验 {exp_id}: {seq} - {dataset} - {strategy}
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment {exp_id} "{config_file}" "{seq}" "{dataset}" "{strategy}" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 {exp_id}"
fi
""")
            exp_id += 1
        
        f.write("""
# 等待所有实验完成
wait_all_experiments

print_separator
print_info "所有实验已完成！"
print_separator

rm -f "${LOG_DIR}"/*.pid

print_info "实验日志位置: ${LOG_DIR}"
print_separator
""")


def _generate_start_script(start_script_path: Path, run_script_path: Path, output_dir: str):
    """生成后台启动脚本"""
    
    log_dir = f"{output_dir}/logs"
    
    with open(start_script_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f"""#!/bin/bash
#===============================================================================
# 全任务序列实验 - 后台启动脚本
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/{run_script_path.name}"
LOG_DIR="{log_dir}"
MASTER_LOG="{log_dir}/master_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
CYAN='\\033[0;36m'
NC='\\033[0m'

echo -e "${{GREEN}}========================================================================${{NC}}"
echo -e "${{GREEN}}全任务序列实验 - 后台启动${{NC}}"
echo -e "${{GREEN}}========================================================================${{NC}}"
echo ""

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo -e "${{YELLOW}}[ERROR]${{NC}} 主脚本不存在: $MAIN_SCRIPT"
    exit 1
fi

echo -e "${{CYAN}}[INFO]${{NC}} 启动主实验脚本..."
echo -e "${{CYAN}}[INFO]${{NC}} 主日志文件: $MASTER_LOG"
echo ""

nohup bash "$MAIN_SCRIPT" > "$MASTER_LOG" 2>&1 &
MASTER_PID=$!

echo $MASTER_PID > "${{LOG_DIR}}/master.pid"

echo -e "${{GREEN}}✓ 主脚本已在后台启动${{NC}}"
echo -e "${{GREEN}}✓ 主进程PID: $MASTER_PID${{NC}}"
echo ""

sleep 3

if ps -p $MASTER_PID > /dev/null; then
    echo -e "${{GREEN}}✓ 实验已成功启动并在后台运行${{NC}}"
    echo ""
    echo -e "${{CYAN}}监控命令:${{NC}}"
    echo -e "  tail -f $MASTER_LOG"
    echo -e "  watch -n 1 nvidia-smi"
    echo ""
    echo -e "${{CYAN}}停止命令:${{NC}}"
    echo -e "  bash $SCRIPT_DIR/stop_all_experiments.sh"
    echo ""
    echo -e "${{GREEN}}现在可以安全地断开SSH连接${{NC}}"
else
    echo -e "${{YELLOW}}[ERROR]${{NC}} 主脚本启动失败，请检查日志: $MASTER_LOG"
    exit 1
fi
""")


def _generate_stop_script(stop_script_path: Path, output_dir: str):
    """生成停止脚本"""
    
    log_dir = f"{output_dir}/logs"
    
    with open(stop_script_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f"""#!/bin/bash
#===============================================================================
# 停止所有全任务序列实验
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

if [ -f "${{LOG_DIR}}/master.pid" ]; then
    MASTER_PID=$(cat "${{LOG_DIR}}/master.pid")
    if ps -p $MASTER_PID > /dev/null; then
        echo -e "${{YELLOW}}停止主进程 (PID: $MASTER_PID)...${{NC}}"
        kill $MASTER_PID
    fi
    rm -f "${{LOG_DIR}}/master.pid"
fi

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


class TaskConfigGenerator:
    """任务配置生成器"""
    
    def __init__(self):
        # 环境配置
        self.environments = {
            "server": {
                "base_dir": "",
                "model_name": "checkpoints/251024/{task}_{dataset}_{strategy}_{seq}.pt",
                "log_dir": "checkpoints/251024/log",
                "checkpoint_dir": "checkpoints/251024",
                "ewc_dir": "checkpoints/251024/ewc_params",
                "gem_dir": "checkpoints/251024/gem_memory"
            },
            "local": {
                "base_dir": "./",
                "model_name": "{task}_{dataset}_{strategy}.pt",
                "log_dir": "./log",
                "checkpoint_dir": "./checkpoints",
                "ewc_dir": "./ewc_params",
                "gem_dir": "./gem_memory"
            },
            "kaggle": {
                "base_dir": "/kaggle/working",
                "model_name": "{task}_{dataset}_{strategy}.pt",
                "log_dir": "/kaggle/working/log",
                "checkpoint_dir": "/kaggle/working/checkpoints",
                "ewc_dir": "/kaggle/working/checkpoints/ewc",
                "gem_dir": "/kaggle/working/checkpoints/gem_memory"
            },
            "autodl": {
                "base_dir": "/root/autodl-tmp",
                "model_name": "{task}_{dataset}_{strategy}.pt",
                "log_dir": "/root/autodl-tmp/log",
                "checkpoint_dir": "/root/autodl-tmp/checkpoints",
                "ewc_dir": "/root/autodl-tmp/checkpoints/ewc",
                "gem_dir": "/root/autodl-tmp/checkpoints/gem_memory"
            }
        }
        
        # 数据集配置 - 针对不同任务使用不同路径
        self.datasets = {
            "twitter2015": {
                "data_dir": "./data",
                "dataset_name": "twitter2015",
                "full_files": {
                    "train": "data/MASC/twitter2015/train.txt",
                    "dev": "data/MASC/twitter2015/dev.txt",
                    "test": "data/MASC/twitter2015/test.txt"
                }
            },
            "twitter2017": {
                "data_dir": "./data",
                "dataset_name": "twitter2017",
                "full_files": {
                    "train": "data/MASC/twitter2017/train.txt",
                    "dev": "data/MASC/twitter2017/dev.txt",
                    "test": "data/MASC/twitter2017/test.txt"
                }
            },
            "mix": {
                "data_dir": "./data",
                "dataset_name": "mix",
                "full_files": {
                    "train": "data/MASC/mix/train.txt",
                    "dev": "data/MASC/mix/dev.txt",
                    "test": "data/MASC/mix/test.txt"
                }
            },
            "200": {
                "data_dir": "./data",
                "dataset_name": "twitter2015",
                "files": {
                    "train": "data/MASC/twitter2015/train__.txt",
                    "dev": "data/MASC/twitter2015/dev__.txt",
                    "test": "data/MASC/twitter2015/test__.txt"
                }
            }
        }
        
        # MNER专用数据集配置
        self.mner_datasets = {
            "twitter2015_ner": {
                "data_dir": "./data",
                "dataset_name": "twitter2015",
                "full_files": {
                    "train": "data/MNER/twitter2015/train.txt",
                    "dev": "data/MNER/twitter2015/dev.txt", 
                    "test": "data/MNER/twitter2015/test.txt"
                }
            },
            "twitter2017_ner": {
                "data_dir": "./data",
                "dataset_name": "twitter2017",
                "full_files": {
                    "train": "data/MNER/twitter2017/train.txt",
                    "dev": "data/MNER/twitter2017/dev.txt",
                    "test": "data/MNER/twitter2017/test.txt"
                }
            },
            "mix_ner": {
                "data_dir": "./data",
                "dataset_name": "mix",
                "full_files": {
                    "train": "data/MNER/mix/train.txt",
                    "dev": "data/MNER/mix/dev.txt",
                    "test": "data/MNER/mix/test.txt"
                }
            },
            "200_ner": {
                "data_dir": "./data",
                "dataset_name": "twitter2015",
                "files": {
                    "train": "data/MNER/twitter2015/train__.txt",
                    "dev": "data/MNER/twitter2015/dev__.txt",
                    "test": "data/MNER/twitter2015/test__.txt"
                }
            }
        }
        
        
        # 持续学习策略配置
        self.strategies = {
            "none": {
                "params": {},
                "description": "无持续学习策略"
            },
            "ewc": {
                "params": {
                    "ewc": 1,
                    "ewc_lambda": 1000.0
                },
                "description": "Elastic Weight Consolidation"
            },
            "replay": {
                "params": {
                    "replay": 1,
                    "memory_percentage": 0.05,
                    "replay_ratio": 0.5,
                    "replay_frequency": 4
                },
                "description": "Experience Replay"
            },
            "lwf": {
                "params": {
                    "lwf": 1,
                    "lwf_T": 2.0,
                    "lwf_alpha": 0.5,
                    "lwf_decay": 0.5
                },
                "description": "Learning without Forgetting"
            },
            "si": {
                "params": {
                    "si": 1,
                    "si_epsilon": 0.1,
                    "si_decay": 0.5
                },
                "description": "Synaptic Intelligence"
            },
            "mas": {
                "params": {
                    "mas": 1,
                    "mas_eps": 1e-3,
                    "mas_decay": 0.5
                },
                "description": "Memory Aware Synapses"
            },
            "gem": {
                "params": {
                    "gem": 1,
                    "gem_mem": 100
                },
                "description": "Gradient Episodic Memory"
            },
            "mymethod": {
                "params": {
                    "mymethod": 1,
                    "ewc_lambda": 1000.0
                },
                "description": "自定义方法"
            },
            "tam_cl": {
                "params": {
                    "tam_cl": 1
                },
                "description": "TAM-CL"
            },
            "moe": {
                "params": {
                    "moe_adapters": 1,
                    "moe_num_experts": 4,
                    "moe_top_k": 2,
                    "ddas": 1
                },
                "description": "MoE Adapters"
            },
            "clap4clip": {
                "params": {
                    "clap4clip": 1,
                    "adapter_size": 64,
                    "finetune_lambda": 0.1,
                    "temperature": 0.07
                },
                "description": "CLAP4CLIP with Adapters and Probabilistic Finetuning"
            },
            "deqa": {
                "params": {
                    "deqa": 1,
                    "deqa_use_description": True,
                    "deqa_use_clip": True,
                    "deqa_ensemble_method": "weighted",
                    "deqa_freeze_old_experts": True,
                    "deqa_distill_weight": 0.5
                },
                "description": "DEQA - Descriptions Enhanced Question-Answering Framework"
            }
        }
    
    def get_dataset_files(self, task_name, dataset):
        """
        获取数据集文件路径
        
        注意：
        - MASC/MATE/MABSA 共享 data/MASC/ 下的文件
        - MNER 使用 data/MNER/ 下的文件
        - 所有任务的图像都在 data/img
        """
        # MNER 任务单独处理
        if task_name == "mner":
            if dataset == "200":
                return self.mner_datasets["200_ner"]["files"]
            elif dataset == "twitter2015":
                return self.mner_datasets["twitter2015_ner"]["full_files"]
            elif dataset == "twitter2017":
                return self.mner_datasets["twitter2017_ner"]["full_files"]
            elif dataset == "mix":
                return self.mner_datasets["mix_ner"]["full_files"]
            else:
                raise ValueError(f"Unknown dataset for mner: {dataset}")
        # MASC/MATE/MABSA 都使用 MASC 文件夹下的数据
        else:
            if dataset == "200":
                return self.datasets["200"]["files"]
            elif dataset in ["twitter2015", "twitter2017", "mix"]:
                return self.datasets[dataset]["full_files"]
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
    
    def determine_mode_suffix(self, modes: List[str]) -> str:
        """根据模式列表确定文件后缀"""
        if not modes:
            return ""
        
        unique_modes = set(modes)
        if len(unique_modes) == 1:
            if "text_only" in unique_modes:
                return "t"
            elif "multimodal" in unique_modes:
                return "m"
            else:
                return ""
        else:
            # 既有text_only又有multimodal
            if "text_only" in unique_modes and "multimodal" in unique_modes:
                return "t2m"
            else:
                return ""
    
    def generate_file_names(self, env: str, dataset: str, strategy: str, modes: List[str], 
                           use_label_embedding: bool = False, seq_suffix: str = "") -> Dict[str, str]:
        """生成文件名称"""
        env_config = self.environments[env]
        
        # 确定模式后缀
        mode_suffix = self.determine_mode_suffix(modes)
        
        # 基础名称
        base_name = f"{dataset}_{strategy}"
        if mode_suffix:
            base_name += f"_{mode_suffix}"
        if use_label_embedding:
            base_name += "_label_emb"
        if seq_suffix:
            base_name += f"_{seq_suffix}"
        
        # 模型名称 - 修复：统一使用base_name，不再硬编码为1.pt
        if env == "server":
            model_name = f"{base_name}.pt"
        else:
            model_name = f"model_{base_name}.pt"
        
        # 训练信息JSON
        train_info_json = f"train_info_{base_name}.json"
        
        # 任务头文件
        task_heads_name = f"model_{base_name}_task_heads.pt"
        
        # 标签嵌入文件
        label_embedding_name = f"label_embedding_{base_name}.pt"
        
        return {
            "model_name": model_name,
            "train_info_json": train_info_json,
            "task_heads_name": task_heads_name,
            "label_embedding_name": label_embedding_name,
            "base_name": base_name,
            "mode_suffix": mode_suffix
        }
    
    def create_task_config(self, task_name: str, session_name: str, dataset: str, 
                          env: str, strategy: str, mode: str, **kwargs) -> Dict[str, Any]:
        """创建单个任务的配置"""
        
        # 获取数据集文件
        dataset_files = self.get_dataset_files(task_name, dataset)
        
        # 为DEQA策略设置description_file
        description_file = None
        if strategy == "deqa":
            if dataset == "twitter2015":
                description_file = "reference/DEQA/DEQA/datasets/release/twitter2015/description_roberta.jsonl"
            elif dataset == "twitter2017":
                description_file = "reference/DEQA/DEQA/datasets/release/twitter2017/description_roberta.jsonl"
            elif dataset == "mix":
                description_file = "reference/DEQA/DEQA/datasets/release/mix/description_roberta.jsonl"
        
        # 任务特定参数
        task_specific_params = {
            "masc": {"num_labels": 3, "epochs": 20, "lr": 5e-6, "step_size": 15, "gamma": 0.5},
            "mate": {"num_labels": 3, "epochs": 20, "lr": 5e-5, "step_size": 10, "gamma": 0.5},
            "mner": {"num_labels": 9, "epochs": 20, "lr": 5e-5, "step_size": 10, "gamma": 0.5},
            "mabsa": {"num_labels": 7, "epochs": 20, "lr": 5e-5, "step_size": 10, "gamma": 0.5}
        }
        
        # 如果是200数据集，所有任务的epoch都改为1
        if dataset == "200":
            for task_key in task_specific_params:
                task_specific_params[task_key]["epochs"] = 1
        
        # 合并参数
        task_params = {**task_specific_params.get(task_name, {}), **kwargs}
        
        # 策略参数
        strategy_params = self.strategies.get(strategy, {}).get("params", {})
        
        # 环境配置
        env_config = self.environments[env]

        use_label_embedding = task_params.get("use_label_embedding", False) or kwargs.get("use_label_embedding", False)
        
        # if use_label_embedding:
        #     task_params["lr"] = 1e-3
        #     task_params["weight_decay"] = 1e-4
        #     task_params["gamma"] = 0.7
            # 当使用label_embedding时，自动启用混合头
            # task_params["use_hierarchical_head"] = True
            
        base_config = {
            "task_name": task_name,
            "session_name": session_name,
            "dataset": dataset,
            "env": env,
            "strategy": strategy,
            "mode": mode,
            # 模型参数
            "num_labels": task_params.get("num_labels", 3),
            "epochs": task_params.get("epochs", 20),
            "lr": task_params.get("lr", 5e-5),
            "batch_size": task_params.get("batch_size", 8),
            "step_size": task_params.get("step_size", 10),
            "gamma": task_params.get("gamma", 0.5),
            "weight_decay": task_params.get("weight_decay", 1e-5),
            "dropout_prob": task_params.get("dropout_prob", 0.3),
            "patience": task_params.get("patience", 999),  # 实际上禁用早停（超参数搜索需要完整训练）
            "fusion_strategy": task_params.get("fusion_strategy", "concat"),
            "num_heads": task_params.get("num_heads", 8),
            "hidden_dim": task_params.get("hidden_dim", 768),
            "text_model_name": task_params.get("text_model_name", "microsoft/deberta-v3-base" if strategy != "clap4clip" else "openai/clip-vit-base-patch32"),
            "image_model_name": task_params.get("image_model_name", "google/vit-base-patch16-224-in21k" if strategy != "clap4clip" else "openai/clip-vit-base-patch32"),
            "image_dir": task_params.get("image_dir", "data/img"),  # 所有任务的图像都在data/img
            # 数据集文件
            "train_text_file": dataset_files["train"],
            "test_text_file": dataset_files["test"],
            "dev_text_file": dataset_files["dev"],
            # 持续学习策略参数
            **strategy_params,
            # DEQA描述文件
            "description_file": description_file,
            # 标签嵌入
            "use_label_embedding": task_params.get("use_label_embedding", False),
            "use_hierarchical_head": task_params.get("use_hierarchical_head", False),
            "label_emb_dim": task_params.get("label_emb_dim", 128),
            "use_similarity_reg": task_params.get("use_similarity_reg", False),
            "similarity_weight": task_params.get("similarity_weight", 0.1),
            # 模型头部参数
            "triaffine": task_params.get("triaffine", 1),
            "span_hidden": task_params.get("span_hidden", 256),
            # CRF和Span Loss参数
            "use_crf": task_params.get("use_crf", 1),  # 默认启用CRF
            "use_span_loss": task_params.get("use_span_loss", 1),  # 默认启用Span Loss
            "boundary_weight": task_params.get("boundary_weight", 0.2),
            "span_f1_weight": task_params.get("span_f1_weight", 0.0),
            "transition_weight": task_params.get("transition_weight", 0.0),
            # 图平滑参数
            "graph_smooth": task_params.get("graph_smooth", 1),
            "graph_tau": task_params.get("graph_tau", 0.5),
            # BiLSTM参数（新增）
            "use_bilstm": task_params.get("use_bilstm", 1),  # 默认使用BiLSTM
            "bilstm_hidden_size": task_params.get("bilstm_hidden_size", 256),
            "bilstm_num_layers": task_params.get("bilstm_num_layers", 2),
            "lstm_lr": task_params.get("lstm_lr", 1e-4),
            "crf_lr": task_params.get("crf_lr", 1e-3),
            # 其他参数
            "num_workers": task_params.get("num_workers", 4),
        }
        
        return base_config
    
    def generate_task_sequence_config(self, env: str, dataset: str, 
                                    task_sequence: List[str] = None,
                                    mode_sequence: List[str] = None,
                                    strategy: str = "none",
                                    use_label_embedding: bool = False,
                                    seq_suffix: str = "",
                                    **kwargs) -> Dict[str, Any]:
        """生成完整的任务序列配置"""
        
        if task_sequence is None:
            task_sequence = ["masc", "mate", "mner", "mabsa"]
        
        if mode_sequence is None:
            # 默认所有任务都使用multimodal模式
            mode_sequence = ["multimodal"] * len(task_sequence)
        
        # 确保任务序列和模式序列长度一致
        if len(task_sequence) != len(mode_sequence):
            raise ValueError(f"任务序列长度({len(task_sequence)})与模式序列长度({len(mode_sequence)})不匹配")
        
        # 环境配置
        env_config = self.environments[env]
        
        # 文件名称（基于所有模式）
        file_names = self.generate_file_names(env, dataset, strategy, mode_sequence, use_label_embedding, seq_suffix)
        
        # 创建任务配置列表
        tasks = []
        for i, (task_name, mode) in enumerate(zip(task_sequence, mode_sequence)):
            session_name = f"{task_name}_{i+1}"
            
            # 创建任务配置
            task_config = self.create_task_config(
                task_name=task_name,
                session_name=session_name,
                dataset=dataset,
                env=env,
                strategy=strategy,
                mode=mode,
                use_label_embedding=use_label_embedding,
                **kwargs
            )
            
            # 设置标签嵌入路径
            if task_config.get("use_label_embedding", False):
                task_config["label_embedding_path"] = f"{env_config['checkpoint_dir']}/{file_names['label_embedding_name']}"
            
            tasks.append(task_config)
        
        # 创建完整配置
        config = {
            "env": env,
            "dataset": dataset,
            "strategy": strategy,
            "mode_sequence": mode_sequence,
            "mode_suffix": file_names["mode_suffix"],
            "use_label_embedding": use_label_embedding,
            "seq_suffix": seq_suffix,
            "total_tasks": len(tasks),
            "tasks": tasks,
            "global_params": {
                "base_dir": env_config["base_dir"],
                "output_model_path": f"{env_config['checkpoint_dir']}/{file_names['model_name']}",
                "train_info_json": f"{env_config['checkpoint_dir']}/{file_names['train_info_json']}",
                "task_heads_path": f"{env_config['checkpoint_dir']}/{file_names['task_heads_name']}",
                "label_embedding_path": f"{env_config['checkpoint_dir']}/{file_names['label_embedding_name']}",
                "ewc_dir": env_config["ewc_dir"],
                "gem_mem_dir": env_config["gem_dir"],
                "log_dir": env_config["log_dir"],
                "checkpoint_dir": env_config["checkpoint_dir"],
                "num_workers": 4,
                "data_dir": "./data",
                "dataset_name": dataset,
            }
        }
        
        return config


def generate_all_task_configs(env: str = "server",
                             datasets: List[str] = None,
                             strategies: List[str] = None,
                             sequences: List[str] = None,
                             task_sequence: List[str] = None,
                             mode_sequence: List[str] = None,
                             output_dir: str = "scripts/configs/all_task"):
    """
    生成所有任务配置文件
    
    Args:
        env: 环境（server或local）
        datasets: 数据集列表
        strategies: 策略列表
        sequences: 序列后缀列表（如seq1, seq2）
        task_sequence: 任务序列
        mode_sequence: 模式序列
        output_dir: 输出目录
    """
    if datasets is None:
        datasets = ["twitter2015", "twitter2017", "mix"]
    
    if strategies is None:
        # 注意顺序：DEQA -> NONE -> MOE -> REPLAY -> 其他
        strategies = ["deqa", "none", "moe", "replay", "ewc", "lwf", "si", "mas", "gem"]
    
    if sequences is None:
        sequences = ["seq1", "seq2"]
    
    if task_sequence is None:
        # 默认8个任务：4个text_only + 4个multimodal
        task_sequence = ["masc", "mate", "mner", "mabsa", "masc", "mate", "mner", "mabsa"]
    
    if mode_sequence is None:
        mode_sequence = ["text_only", "text_only", "text_only", "text_only",
                        "multimodal", "multimodal", "multimodal", "multimodal"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = TaskConfigGenerator()
    configs_generated = []
    
    print(f"生成配置文件到: {output_path}")
    print(f"序列数量: {len(sequences)}")
    print(f"数据集数量: {len(datasets)}")
    print(f"策略数量: {len(strategies)}")
    print(f"总配置文件数: {len(sequences) * len(datasets) * len(strategies)}")
    print()
    
    # 按照指定顺序生成配置
    for seq in sequences:
        for dataset in datasets:
            for strategy in strategies:
                # 生成配置
                config = generator.generate_task_sequence_config(
                    env=env,
                    dataset=dataset,
                    task_sequence=task_sequence,
                    mode_sequence=mode_sequence,
                    strategy=strategy,
                    use_label_embedding=False,
                    seq_suffix=seq
                )
                
                # 生成文件名
                config_name = f"{env}_{dataset}_{strategy}_{seq}.json"
                config_file = output_path / config_name
                
                # 保存配置
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                configs_generated.append({
                    "file": config_file.as_posix(),
                    "seq": seq,
                    "dataset": dataset,
                    "strategy": strategy,
                    "env": env
                })
                
                print(f"✓ {config_name}")
    
    # 生成索引文件
    index_file = output_path / "config_index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_configs": len(configs_generated),
            "sequences": sequences,
            "datasets": datasets,
            "strategies": strategies,
            "task_sequence": task_sequence,
            "mode_sequence": mode_sequence,
            "configs": configs_generated
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n索引文件已生成: {index_file}")
    print(f"\n总共生成 {len(configs_generated)} 个配置文件")
    
    return configs_generated


def main():
    parser = argparse.ArgumentParser(description="生成任务配置文件")
    parser.add_argument("--env", type=str, default="server", 
                       choices=["local", "server", "kaggle", "autodl"],
                       help="环境类型")
    parser.add_argument("--dataset", type=str, default="200", 
                       choices=["twitter2015", "twitter2017", "mix", "200"],
                       help="数据集名称")
    parser.add_argument("--strategy", type=str, default="none",
                       choices=["none", "ewc", "replay", "lwf", "si", "mas", "gem", "mymethod", "tam_cl", "moe", "clap4clip", "deqa"],
                       help="持续学习策略")
    parser.add_argument("--task_sequence", type=str, nargs="+", 
                       default=["masc", "mate", "mner", "mabsa", "masc", "mate", "mner", "mabsa"],
                       help="任务序列")
    parser.add_argument("--mode_sequence", type=str, nargs="+", 
                       default=["text_only", "text_only", "text_only", "text_only", "multimodal", "multimodal", "multimodal", "multimodal"],
                       help="模式序列（与任务序列一一对应）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出配置文件路径（可选，会自动生成）")
    parser.add_argument("--use_label_embedding", action="store_true",
                       help="是否使用标签嵌入")
    parser.add_argument("--seq_suffix", type=str, default="",
                       help="序列后缀（如seq1、seq2等）")
    parser.add_argument("--generate_all", action="store_true",
                       help="生成所有任务配置文件和运行脚本")
    parser.add_argument("--output_dir", type=str, default="scripts/configs/all_task",
                       help="输出目录（用于--generate_all）")
    
    args = parser.parse_args()
    
    # 如果指定了--generate_all，生成所有配置和运行脚本
    if args.generate_all:
        configs = generate_all_task_configs(
            env=args.env,
            output_dir=args.output_dir
        )
        
        # 生成运行脚本
        _generate_run_scripts(args.output_dir, configs)
        
        print(f"\n✓ 运行脚本已生成")
        print(f"✓ 后台启动脚本已生成")
        print(f"✓ 停止脚本已生成")
        print(f"\n使用方法：")
        print(f"  bash {args.output_dir}/start_all_experiments.sh")
        return
    
    # 创建配置生成器
    generator = TaskConfigGenerator()
    
    # 生成配置
    config = generator.generate_task_sequence_config(
        env=args.env,
        dataset=args.dataset,
        task_sequence=args.task_sequence,
        mode_sequence=args.mode_sequence,
        strategy=args.strategy,
        use_label_embedding=args.use_label_embedding,
        seq_suffix=args.seq_suffix
    )
    
    # 自动生成文件名
    if args.output is None:
        # 创建configs目录
        configs_dir = Path("scripts/configs")
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        filename_parts = [args.env, args.dataset, args.strategy]
        if args.use_label_embedding:
            filename_parts.append("label_emb")
        filename = "_".join(filename_parts) + ".json"
        
        output_path = configs_dir / filename
    else:
        output_path = Path(args.output)
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"任务配置文件已生成: {output_path}")
    print(f"环境: {args.env}")
    print(f"数据集: {args.dataset}")
    print(f"策略: {args.strategy}")
    print(f"模式后缀: {config['mode_suffix']}")
    print(f"标签嵌入: {'是' if args.use_label_embedding else '否'}")
    print(f"包含 {len(config['tasks'])} 个任务:")
    
    for i, (task, mode) in enumerate(zip(config['tasks'], config['mode_sequence'])):
        print(f"  {i+1}. {task['task_name']} ({task['session_name']}) - {mode}")
        print(f"     数据集: {task['train_text_file']}")
        print(f"     标签数: {task['num_labels']}")
        print(f"     训练轮数: {task['epochs']}")
        print(f"     学习率: {task['lr']}")
        print(f"     批次大小: {task['batch_size']}")
        print()
    
    # 显示文件路径
    print(f"\n文件路径:")
    model_path = config['global_params']['output_model_path'].replace('\\', '/')
    train_info_path = config['global_params']['train_info_json'].replace('\\', '/')
    task_heads_path = config['global_params']['task_heads_path'].replace('\\', '/')
    print(f"  模型文件: {model_path}")
    print(f"  训练信息: {train_info_path}")
    print(f"  任务头文件: {task_heads_path}")
    if args.use_label_embedding:
        label_emb_path = config['global_params']['label_embedding_path'].replace('\\', '/')
        print(f"  标签嵌入: {label_emb_path}")
    
    # 显示使用示例
    print(f"\n使用示例:")
    # 将路径中的反斜杠转换为正斜杠，确保跨平台兼容
    config_path = str(output_path).replace('\\', '/')
    print(f"python -m scripts.train_with_zero_shot --config {config_path}")


if __name__ == "__main__":
    main() 