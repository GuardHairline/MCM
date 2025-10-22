#!/bin/bash
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "项目根目录: $PROJECT_ROOT"
echo "当前工作目录: $(pwd)"

# 创建日志目录
LOG_DIR="scripts/configs/all_task/logs"
mkdir -p "$LOG_DIR"

# 获取时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

#===============================================================================
# 辅助函数（GPU检测逻辑）
#===============================================================================

print_separator() {
    echo -e "${BLUE}===============================================================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_gpu_info() {
    echo -e "${CYAN}[GPU]${NC} $1"
}

detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        echo $gpu_count
    else
        print_warning "nvidia-smi未找到，假设有1个GPU"
        echo 1
    fi
}

get_free_gpu() {
    if command -v nvidia-smi &> /dev/null; then
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

wait_for_free_gpu() {
    local max_wait=${1:-3600}
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
        
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $free_gpu 2>/dev/null | sed 's/,//g' | awk '{print $1}')
        
        if [ -z "$gpu_util" ]; then
            echo "$free_gpu"
            return 0
        fi
        
        # 如果GPU使用率 < 10%，进行连续检测确认
        if [ "$gpu_util" -lt 10 ] 2>/dev/null; then
            print_info "发现GPU${free_gpu}使用率较低(${gpu_util}%)，开始连续检测验证..." >&2
            
            local continuous_check_duration=60
            local continuous_check_interval=5
            local checks_needed=$((continuous_check_duration / continuous_check_interval))
            local checks_passed=0
            
            for ((i=1; i<=checks_needed; i++)); do
                local current_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $free_gpu 2>/dev/null | sed 's/,//g' | awk '{print $1}')
                local mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $free_gpu 2>/dev/null | sed 's/,//g' | awk '{print $1}')
                local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $free_gpu 2>/dev/null | sed 's/,//g' | awk '{print $1}')
                
                local mem_percent=0
                if [ -n "$mem_total" ] && [ "$mem_total" -gt 0 ] 2>/dev/null; then
                    mem_percent=$((mem_used * 100 / mem_total))
                fi
                
                # 连续检测标准：使用率 < 10% 且 显存 < 10%
                if [ "$current_util" -lt 10 ] 2>/dev/null && [ "$mem_percent" -lt 10 ] 2>/dev/null; then
                    checks_passed=$((checks_passed + 1))
                    print_info "  检查 $i/${checks_needed}: ✓ 通过 (使用率: ${current_util}%, 显存: ${mem_percent}%)" >&2
                else
                    print_warning "  检查 $i/${checks_needed}: ✗ 未通过 (使用率: ${current_util}%, 显存: ${mem_percent}%)，放弃此GPU" >&2
                    break
                fi
                
                if [ $i -lt $checks_needed ]; then
                    sleep $continuous_check_interval
                fi
            done
            
            # 如果连续检测通过，使用该GPU
            if [ $checks_passed -eq $checks_needed ]; then
                print_info "GPU${free_gpu}连续检测通过（使用率<10%且显存<10%），可以使用" >&2
                echo "$free_gpu"
                return 0
            else
                print_warning "GPU${free_gpu}连续检测未通过，继续等待..." >&2
            fi
        fi
        
        print_warning "所有GPU都在使用中 (GPU$free_gpu: ${gpu_util}%)，等待 ${check_interval} 秒后重试..." >&2
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
        
        print_info "正在对GPU${fallback_gpu}进行连续检测（需要连续${checks_needed}次通过）..." >&2
        
        for ((i=1; i<=checks_needed; i++)); do
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $fallback_gpu 2>/dev/null | sed 's/,//g' | awk '{print $1}')
            local mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $fallback_gpu 2>/dev/null | sed 's/,//g' | awk '{print $1}')
            local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $fallback_gpu 2>/dev/null | sed 's/,//g' | awk '{print $1}')
            
            local mem_percent=0
            if [ -n "$mem_total" ] && [ "$mem_total" -gt 0 ] 2>/dev/null; then
                mem_percent=$((mem_used * 100 / mem_total))
            fi
            
            # 宽松检测标准：使用率 < 20% 且 显存 < 20%
            if [ "$gpu_util" -lt 20 ] 2>/dev/null && [ "$mem_percent" -lt 20 ] 2>/dev/null; then
                checks_passed=$((checks_passed + 1))
                print_info "  检查 $i/${checks_needed}: ✓ 通过 (使用率: ${gpu_util}%, 显存: ${mem_percent}%)" >&2
            else
                print_warning "  检查 $i/${checks_needed}: ✗ 未通过 (使用率: ${gpu_util}%, 显存: ${mem_percent}%)，重新开始检测..." >&2
                break
            fi
            
            if [ $i -lt $checks_needed ]; then
                sleep $continuous_check_interval
            fi
        done
        
        if [ $checks_passed -eq $checks_needed ]; then
            print_info "GPU${fallback_gpu}连续检测通过（使用率<20%且显存<20%），可以使用" >&2
            echo "$fallback_gpu"
            return 0
        fi
        
        print_warning "GPU${fallback_gpu}连续检测未通过，等待30秒后重新检测..." >&2
        sleep 30
    done
}

run_experiment() {
    local exp_id=$1
    local config_file=$2
    local seq=$3
    local dataset=$4
    local strategy=$5
    local gpu_id=$6
    
    local log_file="${LOG_DIR}/exp_${exp_id}_${seq}_${dataset}_${strategy}_${TIMESTAMP}.log"
    local pid_file="${LOG_DIR}/exp_${exp_id}.pid"
    
    print_separator
    print_info "启动实验 #${exp_id}"
    print_info "  序列: ${seq}"
    print_info "  数据集: ${dataset}"
    print_info "  策略: ${strategy}"
    print_gpu_info "  使用GPU: ${gpu_id}"
    print_info "  日志文件: ${log_file}"
    
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # 运行8个任务（0-7）
    nohup python -m scripts.train_with_zero_shot --config "${config_file}" --start_task 0 --end_task 8 > "${log_file}" 2>&1 &
    
    local pid=$!
    echo $pid > "${pid_file}"
    
    print_info "  进程ID: ${pid}"
    print_info "  PID文件: ${pid_file}"
    
    unset CUDA_VISIBLE_DEVICES
    
    sleep 5
    
    if ps -p $pid > /dev/null; then
        print_info "  实验 #${exp_id} 已成功启动"
    else
        print_error "  实验 #${exp_id} 启动失败"
        return 1
    fi
    
    return 0
}

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
            sleep 60
        fi
    done
    
    print_info "所有实验已完成"
}

#===============================================================================
# 主程序
#===============================================================================

print_separator
print_info "全任务序列实验启动"
print_separator

GPU_COUNT=$(detect_gpus)
print_gpu_info "检测到 ${GPU_COUNT} 个GPU"

if command -v nvidia-smi &> /dev/null; then
    print_separator
    nvidia-smi
    print_separator
fi

print_info "开始运行实验..."
print_separator


# 实验 1: seq1 - twitter2015 - deqa
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 1 "scripts/configs/all_task/server_twitter2015_deqa_seq1.json" "seq1" "twitter2015" "deqa" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 1"
fi

# 实验 2: seq1 - twitter2015 - none
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 2 "scripts/configs/all_task/server_twitter2015_none_seq1.json" "seq1" "twitter2015" "none" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 2"
fi

# 实验 3: seq1 - twitter2015 - moe
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 3 "scripts/configs/all_task/server_twitter2015_moe_seq1.json" "seq1" "twitter2015" "moe" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 3"
fi

# 实验 4: seq1 - twitter2015 - replay
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 4 "scripts/configs/all_task/server_twitter2015_replay_seq1.json" "seq1" "twitter2015" "replay" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 4"
fi

# 实验 5: seq1 - twitter2015 - ewc
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 5 "scripts/configs/all_task/server_twitter2015_ewc_seq1.json" "seq1" "twitter2015" "ewc" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 5"
fi

# 实验 6: seq1 - twitter2015 - lwf
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 6 "scripts/configs/all_task/server_twitter2015_lwf_seq1.json" "seq1" "twitter2015" "lwf" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 6"
fi

# 实验 7: seq1 - twitter2015 - si
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 7 "scripts/configs/all_task/server_twitter2015_si_seq1.json" "seq1" "twitter2015" "si" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 7"
fi

# 实验 8: seq1 - twitter2015 - mas
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 8 "scripts/configs/all_task/server_twitter2015_mas_seq1.json" "seq1" "twitter2015" "mas" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 8"
fi

# 实验 9: seq1 - twitter2015 - gem
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 9 "scripts/configs/all_task/server_twitter2015_gem_seq1.json" "seq1" "twitter2015" "gem" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 9"
fi

# 实验 10: seq1 - twitter2017 - deqa
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 10 "scripts/configs/all_task/server_twitter2017_deqa_seq1.json" "seq1" "twitter2017" "deqa" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 10"
fi

# 实验 11: seq1 - twitter2017 - none
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 11 "scripts/configs/all_task/server_twitter2017_none_seq1.json" "seq1" "twitter2017" "none" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 11"
fi

# 实验 12: seq1 - twitter2017 - moe
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 12 "scripts/configs/all_task/server_twitter2017_moe_seq1.json" "seq1" "twitter2017" "moe" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 12"
fi

# 实验 13: seq1 - twitter2017 - replay
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 13 "scripts/configs/all_task/server_twitter2017_replay_seq1.json" "seq1" "twitter2017" "replay" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 13"
fi

# 实验 14: seq1 - twitter2017 - ewc
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 14 "scripts/configs/all_task/server_twitter2017_ewc_seq1.json" "seq1" "twitter2017" "ewc" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 14"
fi

# 实验 15: seq1 - twitter2017 - lwf
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 15 "scripts/configs/all_task/server_twitter2017_lwf_seq1.json" "seq1" "twitter2017" "lwf" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 15"
fi

# 实验 16: seq1 - twitter2017 - si
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 16 "scripts/configs/all_task/server_twitter2017_si_seq1.json" "seq1" "twitter2017" "si" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 16"
fi

# 实验 17: seq1 - twitter2017 - mas
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 17 "scripts/configs/all_task/server_twitter2017_mas_seq1.json" "seq1" "twitter2017" "mas" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 17"
fi

# 实验 18: seq1 - twitter2017 - gem
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 18 "scripts/configs/all_task/server_twitter2017_gem_seq1.json" "seq1" "twitter2017" "gem" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 18"
fi

# 实验 19: seq1 - mix - deqa
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 19 "scripts/configs/all_task/server_mix_deqa_seq1.json" "seq1" "mix" "deqa" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 19"
fi

# 实验 20: seq1 - mix - none
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 20 "scripts/configs/all_task/server_mix_none_seq1.json" "seq1" "mix" "none" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 20"
fi

# 实验 21: seq1 - mix - moe
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 21 "scripts/configs/all_task/server_mix_moe_seq1.json" "seq1" "mix" "moe" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 21"
fi

# 实验 22: seq1 - mix - replay
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 22 "scripts/configs/all_task/server_mix_replay_seq1.json" "seq1" "mix" "replay" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 22"
fi

# 实验 23: seq1 - mix - ewc
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 23 "scripts/configs/all_task/server_mix_ewc_seq1.json" "seq1" "mix" "ewc" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 23"
fi

# 实验 24: seq1 - mix - lwf
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 24 "scripts/configs/all_task/server_mix_lwf_seq1.json" "seq1" "mix" "lwf" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 24"
fi

# 实验 25: seq1 - mix - si
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 25 "scripts/configs/all_task/server_mix_si_seq1.json" "seq1" "mix" "si" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 25"
fi

# 实验 26: seq1 - mix - mas
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 26 "scripts/configs/all_task/server_mix_mas_seq1.json" "seq1" "mix" "mas" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 26"
fi

# 实验 27: seq1 - mix - gem
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 27 "scripts/configs/all_task/server_mix_gem_seq1.json" "seq1" "mix" "gem" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 27"
fi

# 实验 28: seq2 - twitter2015 - deqa
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 28 "scripts/configs/all_task/server_twitter2015_deqa_seq2.json" "seq2" "twitter2015" "deqa" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 28"
fi

# 实验 29: seq2 - twitter2015 - none
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 29 "scripts/configs/all_task/server_twitter2015_none_seq2.json" "seq2" "twitter2015" "none" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 29"
fi

# 实验 30: seq2 - twitter2015 - moe
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 30 "scripts/configs/all_task/server_twitter2015_moe_seq2.json" "seq2" "twitter2015" "moe" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 30"
fi

# 实验 31: seq2 - twitter2015 - replay
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 31 "scripts/configs/all_task/server_twitter2015_replay_seq2.json" "seq2" "twitter2015" "replay" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 31"
fi

# 实验 32: seq2 - twitter2015 - ewc
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 32 "scripts/configs/all_task/server_twitter2015_ewc_seq2.json" "seq2" "twitter2015" "ewc" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 32"
fi

# 实验 33: seq2 - twitter2015 - lwf
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 33 "scripts/configs/all_task/server_twitter2015_lwf_seq2.json" "seq2" "twitter2015" "lwf" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 33"
fi

# 实验 34: seq2 - twitter2015 - si
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 34 "scripts/configs/all_task/server_twitter2015_si_seq2.json" "seq2" "twitter2015" "si" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 34"
fi

# 实验 35: seq2 - twitter2015 - mas
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 35 "scripts/configs/all_task/server_twitter2015_mas_seq2.json" "seq2" "twitter2015" "mas" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 35"
fi

# 实验 36: seq2 - twitter2015 - gem
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 36 "scripts/configs/all_task/server_twitter2015_gem_seq2.json" "seq2" "twitter2015" "gem" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 36"
fi

# 实验 37: seq2 - twitter2017 - deqa
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 37 "scripts/configs/all_task/server_twitter2017_deqa_seq2.json" "seq2" "twitter2017" "deqa" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 37"
fi

# 实验 38: seq2 - twitter2017 - none
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 38 "scripts/configs/all_task/server_twitter2017_none_seq2.json" "seq2" "twitter2017" "none" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 38"
fi

# 实验 39: seq2 - twitter2017 - moe
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 39 "scripts/configs/all_task/server_twitter2017_moe_seq2.json" "seq2" "twitter2017" "moe" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 39"
fi

# 实验 40: seq2 - twitter2017 - replay
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 40 "scripts/configs/all_task/server_twitter2017_replay_seq2.json" "seq2" "twitter2017" "replay" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 40"
fi

# 实验 41: seq2 - twitter2017 - ewc
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 41 "scripts/configs/all_task/server_twitter2017_ewc_seq2.json" "seq2" "twitter2017" "ewc" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 41"
fi

# 实验 42: seq2 - twitter2017 - lwf
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 42 "scripts/configs/all_task/server_twitter2017_lwf_seq2.json" "seq2" "twitter2017" "lwf" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 42"
fi

# 实验 43: seq2 - twitter2017 - si
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 43 "scripts/configs/all_task/server_twitter2017_si_seq2.json" "seq2" "twitter2017" "si" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 43"
fi

# 实验 44: seq2 - twitter2017 - mas
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 44 "scripts/configs/all_task/server_twitter2017_mas_seq2.json" "seq2" "twitter2017" "mas" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 44"
fi

# 实验 45: seq2 - twitter2017 - gem
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 45 "scripts/configs/all_task/server_twitter2017_gem_seq2.json" "seq2" "twitter2017" "gem" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 45"
fi

# 实验 46: seq2 - mix - deqa
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 46 "scripts/configs/all_task/server_mix_deqa_seq2.json" "seq2" "mix" "deqa" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 46"
fi

# 实验 47: seq2 - mix - none
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 47 "scripts/configs/all_task/server_mix_none_seq2.json" "seq2" "mix" "none" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 47"
fi

# 实验 48: seq2 - mix - moe
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 48 "scripts/configs/all_task/server_mix_moe_seq2.json" "seq2" "mix" "moe" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 48"
fi

# 实验 49: seq2 - mix - replay
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 49 "scripts/configs/all_task/server_mix_replay_seq2.json" "seq2" "mix" "replay" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 49"
fi

# 实验 50: seq2 - mix - ewc
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 50 "scripts/configs/all_task/server_mix_ewc_seq2.json" "seq2" "mix" "ewc" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 50"
fi

# 实验 51: seq2 - mix - lwf
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 51 "scripts/configs/all_task/server_mix_lwf_seq2.json" "seq2" "mix" "lwf" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 51"
fi

# 实验 52: seq2 - mix - si
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 52 "scripts/configs/all_task/server_mix_si_seq2.json" "seq2" "mix" "si" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 52"
fi

# 实验 53: seq2 - mix - mas
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 53 "scripts/configs/all_task/server_mix_mas_seq2.json" "seq2" "mix" "mas" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 53"
fi

# 实验 54: seq2 - mix - gem
GPU_ID=$(wait_for_free_gpu)
if [ $? -eq 0 ]; then
    run_experiment 54 "scripts/configs/all_task/server_mix_gem_seq2.json" "seq2" "mix" "gem" $GPU_ID
    
    if [ $GPU_COUNT -gt 1 ]; then
        sleep 10
    else
        wait
    fi
else
    print_error "无法获取空闲GPU，跳过实验 54"
fi

# 等待所有实验完成
wait_all_experiments

print_separator
print_info "所有实验已完成！"
print_separator

rm -f "${LOG_DIR}"/*.pid

print_info "实验日志位置: ${LOG_DIR}"
print_separator
