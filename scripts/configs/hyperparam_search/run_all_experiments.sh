#!/bin/bash
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
#   - 等待超时后检查：如果使用率>50%或显存>50%，继续等待
#   - 避免干扰其他正在运行的程序
#
# 优化策略:
#   2张GPU: 实验自动轮流分配到GPU0和GPU1
#   多张GPU: 充分并行利用所有GPU
#
# 使用方法:
#   bash run_all_experiments.sh
#
# 日志位置:
#   scripts/configs/hyperparam_search/logs/exp_{id}_{timestamp}.log
#===============================================================================

# 设置错误时退出
set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 脚本在 scripts/configs/hyperparam_search/ 下，需要向上3层到达项目根目录
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 打印当前目录便于调试
echo "项目根目录: $PROJECT_ROOT"
echo "当前工作目录: $(pwd)"

# 创建日志目录
LOG_DIR="scripts/configs/hyperparam_search/logs"
mkdir -p "$LOG_DIR"

# 获取时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

#===============================================================================
# 辅助函数
#===============================================================================

# 打印分隔线
print_separator() {
    echo -e "${BLUE}===============================================================================${NC}"
}

# 打印信息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# 打印警告
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 打印错误
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印GPU信息
print_gpu_info() {
    echo -e "${CYAN}[GPU]${NC} $1"
}

# 检测可用GPU数量
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        echo $gpu_count
    else
        print_warning "nvidia-smi未找到，假设有1个GPU"
        echo 1
    fi
}

# 获取空闲GPU（GPU使用率最低的GPU）
get_free_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        # 获取所有GPU的使用率情况，找到使用率最低的
        # CSV格式用逗号分隔，需要正确处理
        local free_gpu=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | \
                        sed 's/,/ /g' | \
                        awk '{print $1, $2}' | \
                        sort -k2 -n | \
                        head -1 | \
                        awk '{print $1}')
        echo "$free_gpu"
    else
        echo "0"
    fi
}

# 等待空闲GPU
wait_for_free_gpu() {
    local max_wait=${1:-3600}  # 默认最多等待1小时
    local wait_time=0
    local check_interval=30  # 每30秒检查一次
    
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
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $free_gpu 2>/dev/null | sed 's/,//g' | awk '{print $1}')
        
        # 如果无法获取GPU使用率，默认认为可用
        if [ -z "$gpu_util" ]; then
            echo "$free_gpu"
            return 0
        fi
        
        # 如果GPU使用率小于10%，认为GPU空闲
        if [ "$gpu_util" -lt 10 ] 2>/dev/null; then
            echo "$free_gpu"
            return 0
        fi
        
        # 输出警告到stderr，不影响返回值
        print_warning "所有GPU都在使用中 (GPU$free_gpu: ${gpu_util}%)，等待 ${check_interval} 秒后重试..." >&2
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    # 超时后检查GPU使用率和显存使用率，如果有程序在用则继续等待
    local fallback_gpu=$(get_free_gpu)
    local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $fallback_gpu 2>/dev/null | sed 's/,//g' | awk '{print $1}')
    local mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $fallback_gpu 2>/dev/null | sed 's/,//g' | awk '{print $1}')
    local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $fallback_gpu 2>/dev/null | sed 's/,//g' | awk '{print $1}')
    
    # 计算显存使用率
    local mem_percent=0
    if [ -n "$mem_total" ] && [ "$mem_total" -gt 0 ] 2>/dev/null; then
        mem_percent=$((mem_used * 100 / mem_total))
    fi
    
    # 如果GPU使用率超过50%或显存使用率超过50%，说明有其他程序在用，继续等待
    if [ "$gpu_util" -gt 50 ] 2>/dev/null || [ "$mem_percent" -gt 50 ] 2>/dev/null; then
        print_warning "等待超时，但GPU${fallback_gpu}仍在被占用 (使用率: ${gpu_util}%, 显存: ${mem_percent}%)，继续等待..." >&2
        # 递归调用继续等待
        wait_for_free_gpu $max_wait
        return $?
    fi
    
    # 否则返回使用率最低的GPU
    print_warning "等待超时（${max_wait}秒），使用率最低的GPU ${fallback_gpu} (使用率: ${gpu_util}%, 显存: ${mem_percent}%)" >&2
    echo "$fallback_gpu"
    return 0
}

# 运行单个实验
run_experiment() {
    local exp_id=$1
    local config_file=$2
    local strategy=$3
    local lr=$4
    local step_size=$5
    local gamma=$6
    local gpu_id=$7
    
    local log_file="${LOG_DIR}/exp_${exp_id}_${strategy}_${TIMESTAMP}.log"
    local pid_file="${LOG_DIR}/exp_${exp_id}.pid"
    
    print_separator
    print_info "启动实验 #${exp_id}"
    print_info "  策略: ${strategy}"
    print_info "  超参数: lr=${lr}, step_size=${step_size}, gamma=${gamma}"
    print_gpu_info "  使用GPU: ${gpu_id}"
    print_info "  日志文件: ${log_file}"
    
    # 设置CUDA_VISIBLE_DEVICES，只使用指定的GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # 使用nohup在后台运行，添加任务范围参数
    nohup python -m scripts.train_with_zero_shot --config "${config_file}" --start_task 0 --end_task 2 > "${log_file}" 2>&1 &
    
    local pid=$!
    echo $pid > "${pid_file}"
    
    print_info "  进程ID: ${pid}"
    print_info "  PID文件: ${pid_file}"
    
    # 取消CUDA_VISIBLE_DEVICES的导出
    unset CUDA_VISIBLE_DEVICES
    
    # 等待一小段时间，确保进程启动
    sleep 5
    
    # 检查进程是否还在运行
    if ps -p $pid > /dev/null; then
        print_info "  实验 #${exp_id} 已成功启动"
    else
        print_error "  实验 #${exp_id} 启动失败"
        return 1
    fi
    
    return 0
}

# 等待所有后台任务完成
wait_all_experiments() {
    print_separator
    print_info "等待所有实验完成..."
    
    local all_done=false
    while [ "$all_done" = false ]; do
        all_done=true
        
        for pid_file in "${LOG_DIR}"/*.pid; do
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
}

#===============================================================================
# 主程序
#===============================================================================

print_separator
print_info "MASC 超参数搜索实验启动"
print_separator

# 检测GPU数量
GPU_COUNT=$(detect_gpus)
print_gpu_info "检测到 ${GPU_COUNT} 个GPU"

# 显示GPU信息
if command -v nvidia-smi &> /dev/null; then
    print_separator
    nvidia-smi
    print_separator
fi

# 开始实验
print_info "开始运行实验..."
print_separator


# 实验 3: none - lr=5e-05, step_size=15, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 3 "scripts/configs/hyperparam_search/server_twitter2015_none_lr5e05_ss15_g0.5.json" "none" 5e-05 15 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 3"
fi

# 实验 4: none - lr=1e-05, step_size=5, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 4 "scripts/configs/hyperparam_search/server_twitter2015_none_lr1e05_ss5_g0.5.json" "none" 1e-05 5 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 4"
fi

# 实验 5: none - lr=1e-05, step_size=10, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 5 "scripts/configs/hyperparam_search/server_twitter2015_none_lr1e05_ss10_g0.5.json" "none" 1e-05 10 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 5"
fi

# 实验 6: none - lr=1e-05, step_size=15, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 6 "scripts/configs/hyperparam_search/server_twitter2015_none_lr1e05_ss15_g0.5.json" "none" 1e-05 15 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 6"
fi

# 实验 7: none - lr=5e-06, step_size=5, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 7 "scripts/configs/hyperparam_search/server_twitter2015_none_lr5e06_ss5_g0.5.json" "none" 5e-06 5 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 7"
fi

# 实验 8: none - lr=5e-06, step_size=10, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 8 "scripts/configs/hyperparam_search/server_twitter2015_none_lr5e06_ss10_g0.5.json" "none" 5e-06 10 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 8"
fi

# 实验 9: none - lr=5e-06, step_size=15, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 9 "scripts/configs/hyperparam_search/server_twitter2015_none_lr5e06_ss15_g0.5.json" "none" 5e-06 15 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 9"
fi

# 实验 10: none - lr=1e-05, step_size=10, gamma=0.3
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 10 "scripts/configs/hyperparam_search/server_twitter2015_none_lr1e05_ss10_g0.3.json" "none" 1e-05 10 0.3 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 10"
fi

# 实验 11: none - lr=1e-05, step_size=10, gamma=0.7
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 11 "scripts/configs/hyperparam_search/server_twitter2015_none_lr1e05_ss10_g0.7.json" "none" 1e-05 10 0.7 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 11"
fi

# 实验 12: none - lr=5e-05, step_size=5, gamma=0.7
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 12 "scripts/configs/hyperparam_search/server_twitter2015_none_lr5e05_ss5_g0.7.json" "none" 5e-05 5 0.7 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 12"
fi

# 实验 13: none - lr=5e-06, step_size=15, gamma=0.3
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 13 "scripts/configs/hyperparam_search/server_twitter2015_none_lr5e06_ss15_g0.3.json" "none" 5e-06 15 0.3 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 13"
fi

# 实验 14: replay - lr=5e-05, step_size=5, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 14 "scripts/configs/hyperparam_search/server_twitter2015_replay_lr5e05_ss5_g0.5.json" "replay" 5e-05 5 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 14"
fi

# 实验 15: replay - lr=5e-05, step_size=10, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 15 "scripts/configs/hyperparam_search/server_twitter2015_replay_lr5e05_ss10_g0.5.json" "replay" 5e-05 10 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 15"
fi

# 实验 16: replay - lr=5e-05, step_size=15, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 16 "scripts/configs/hyperparam_search/server_twitter2015_replay_lr5e05_ss15_g0.5.json" "replay" 5e-05 15 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 16"
fi

# 实验 17: replay - lr=1e-05, step_size=5, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 17 "scripts/configs/hyperparam_search/server_twitter2015_replay_lr1e05_ss5_g0.5.json" "replay" 1e-05 5 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 17"
fi

# 实验 18: replay - lr=1e-05, step_size=10, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 18 "scripts/configs/hyperparam_search/server_twitter2015_replay_lr1e05_ss10_g0.5.json" "replay" 1e-05 10 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 18"
fi

# 实验 19: replay - lr=1e-05, step_size=15, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 19 "scripts/configs/hyperparam_search/server_twitter2015_replay_lr1e05_ss15_g0.5.json" "replay" 1e-05 15 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 19"
fi

# 实验 20: replay - lr=5e-06, step_size=5, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 20 "scripts/configs/hyperparam_search/server_twitter2015_replay_lr5e06_ss5_g0.5.json" "replay" 5e-06 5 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 20"
fi

# 实验 21: replay - lr=5e-06, step_size=10, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 21 "scripts/configs/hyperparam_search/server_twitter2015_replay_lr5e06_ss10_g0.5.json" "replay" 5e-06 10 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 21"
fi

# 实验 22: replay - lr=5e-06, step_size=15, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 22 "scripts/configs/hyperparam_search/server_twitter2015_replay_lr5e06_ss15_g0.5.json" "replay" 5e-06 15 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 22"
fi

# 实验 23: replay - lr=1e-05, step_size=10, gamma=0.3
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 23 "scripts/configs/hyperparam_search/server_twitter2015_replay_lr1e05_ss10_g0.3.json" "replay" 1e-05 10 0.3 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 23"
fi

# 实验 24: replay - lr=1e-05, step_size=10, gamma=0.7
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 24 "scripts/configs/hyperparam_search/server_twitter2015_replay_lr1e05_ss10_g0.7.json" "replay" 1e-05 10 0.7 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 24"
fi

# 实验 25: replay - lr=5e-05, step_size=5, gamma=0.7
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 25 "scripts/configs/hyperparam_search/server_twitter2015_replay_lr5e05_ss5_g0.7.json" "replay" 5e-05 5 0.7 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 25"
fi

# 实验 26: replay - lr=5e-06, step_size=15, gamma=0.3
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 26 "scripts/configs/hyperparam_search/server_twitter2015_replay_lr5e06_ss15_g0.3.json" "replay" 5e-06 15 0.3 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 26"
fi

# 实验 27: moe - lr=5e-05, step_size=5, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 27 "scripts/configs/hyperparam_search/server_twitter2015_moe_lr5e05_ss5_g0.5.json" "moe" 5e-05 5 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 27"
fi

# 实验 28: moe - lr=5e-05, step_size=10, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 28 "scripts/configs/hyperparam_search/server_twitter2015_moe_lr5e05_ss10_g0.5.json" "moe" 5e-05 10 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 28"
fi

# 实验 29: moe - lr=5e-05, step_size=15, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 29 "scripts/configs/hyperparam_search/server_twitter2015_moe_lr5e05_ss15_g0.5.json" "moe" 5e-05 15 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 29"
fi

# 实验 30: moe - lr=1e-05, step_size=5, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 30 "scripts/configs/hyperparam_search/server_twitter2015_moe_lr1e05_ss5_g0.5.json" "moe" 1e-05 5 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 30"
fi

# 实验 31: moe - lr=1e-05, step_size=10, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 31 "scripts/configs/hyperparam_search/server_twitter2015_moe_lr1e05_ss10_g0.5.json" "moe" 1e-05 10 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 31"
fi

# 实验 32: moe - lr=1e-05, step_size=15, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 32 "scripts/configs/hyperparam_search/server_twitter2015_moe_lr1e05_ss15_g0.5.json" "moe" 1e-05 15 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 32"
fi

# 实验 33: moe - lr=5e-06, step_size=5, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 33 "scripts/configs/hyperparam_search/server_twitter2015_moe_lr5e06_ss5_g0.5.json" "moe" 5e-06 5 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 33"
fi

# 实验 34: moe - lr=5e-06, step_size=10, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 34 "scripts/configs/hyperparam_search/server_twitter2015_moe_lr5e06_ss10_g0.5.json" "moe" 5e-06 10 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 34"
fi

# 实验 35: moe - lr=5e-06, step_size=15, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 35 "scripts/configs/hyperparam_search/server_twitter2015_moe_lr5e06_ss15_g0.5.json" "moe" 5e-06 15 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 35"
fi

# 实验 36: moe - lr=1e-05, step_size=10, gamma=0.3
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 36 "scripts/configs/hyperparam_search/server_twitter2015_moe_lr1e05_ss10_g0.3.json" "moe" 1e-05 10 0.3 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 36"
fi

# 实验 37: moe - lr=1e-05, step_size=10, gamma=0.7
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 37 "scripts/configs/hyperparam_search/server_twitter2015_moe_lr1e05_ss10_g0.7.json" "moe" 1e-05 10 0.7 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 37"
fi

# 实验 38: moe - lr=5e-05, step_size=5, gamma=0.7
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 38 "scripts/configs/hyperparam_search/server_twitter2015_moe_lr5e05_ss5_g0.7.json" "moe" 5e-05 5 0.7 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 38"
fi

# 实验 39: moe - lr=5e-06, step_size=15, gamma=0.3
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 39 "scripts/configs/hyperparam_search/server_twitter2015_moe_lr5e06_ss15_g0.3.json" "moe" 5e-06 15 0.3 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 39"
fi

# 实验 40: deqa - lr=5e-05, step_size=5, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 40 "scripts/configs/hyperparam_search/server_twitter2015_deqa_lr5e05_ss5_g0.5.json" "deqa" 5e-05 5 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 40"
fi

# 实验 41: deqa - lr=5e-05, step_size=10, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 41 "scripts/configs/hyperparam_search/server_twitter2015_deqa_lr5e05_ss10_g0.5.json" "deqa" 5e-05 10 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 41"
fi

# 实验 42: deqa - lr=5e-05, step_size=15, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 42 "scripts/configs/hyperparam_search/server_twitter2015_deqa_lr5e05_ss15_g0.5.json" "deqa" 5e-05 15 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 42"
fi

# 实验 43: deqa - lr=1e-05, step_size=5, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 43 "scripts/configs/hyperparam_search/server_twitter2015_deqa_lr1e05_ss5_g0.5.json" "deqa" 1e-05 5 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 43"
fi

# 实验 44: deqa - lr=1e-05, step_size=10, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 44 "scripts/configs/hyperparam_search/server_twitter2015_deqa_lr1e05_ss10_g0.5.json" "deqa" 1e-05 10 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 44"
fi

# 实验 45: deqa - lr=1e-05, step_size=15, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 45 "scripts/configs/hyperparam_search/server_twitter2015_deqa_lr1e05_ss15_g0.5.json" "deqa" 1e-05 15 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 45"
fi

# 实验 46: deqa - lr=5e-06, step_size=5, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 46 "scripts/configs/hyperparam_search/server_twitter2015_deqa_lr5e06_ss5_g0.5.json" "deqa" 5e-06 5 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 46"
fi

# 实验 47: deqa - lr=5e-06, step_size=10, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 47 "scripts/configs/hyperparam_search/server_twitter2015_deqa_lr5e06_ss10_g0.5.json" "deqa" 5e-06 10 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 47"
fi

# 实验 48: deqa - lr=5e-06, step_size=15, gamma=0.5
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 48 "scripts/configs/hyperparam_search/server_twitter2015_deqa_lr5e06_ss15_g0.5.json" "deqa" 5e-06 15 0.5 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 48"
fi

# 实验 49: deqa - lr=1e-05, step_size=10, gamma=0.3
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 49 "scripts/configs/hyperparam_search/server_twitter2015_deqa_lr1e05_ss10_g0.3.json" "deqa" 1e-05 10 0.3 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 49"
fi

# 实验 50: deqa - lr=1e-05, step_size=10, gamma=0.7
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 50 "scripts/configs/hyperparam_search/server_twitter2015_deqa_lr1e05_ss10_g0.7.json" "deqa" 1e-05 10 0.7 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 50"
fi

# 实验 51: deqa - lr=5e-05, step_size=5, gamma=0.7
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 51 "scripts/configs/hyperparam_search/server_twitter2015_deqa_lr5e05_ss5_g0.7.json" "deqa" 5e-05 5 0.7 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 51"
fi

# 实验 52: deqa - lr=5e-06, step_size=15, gamma=0.3
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 52 "scripts/configs/hyperparam_search/server_twitter2015_deqa_lr5e06_ss15_g0.3.json" "deqa" 5e-06 15 0.3 $GPU_ID
    
    # 如果有多个GPU，给下一个实验一些启动时间
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        # 单GPU情况下，等待当前实验完成
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 52"
fi

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
