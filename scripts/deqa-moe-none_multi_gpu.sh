#!/bin/bash
#===============================================================================
# Twitter2015 多策略对比实验脚本 - 多GPU优化版
# 
# 功能:
#   - 自动检测可用GPU数量
#   - 智能分配实验到不同GPU（充分利用资源）
#   - 支持SSH断开后继续运行 (使用nohup)
#   - 独立日志文件
#   - GPU使用监控
#
# 优化策略:
#   2张GPU: DEQA在GPU0, MoE在GPU1, None等待前面完成后运行
#   3张GPU: DEQA在GPU0, MoE在GPU1, None在GPU2（完全并行）
#
# 使用方法:
#   bash scripts/deqa-moe-none_multi_gpu.sh
#
# 日志位置:
#   logs/twitter2015/deqa_seq1_YYYYMMDD_HHMMSS.log
#   logs/twitter2015/moe_seq1_YYYYMMDD_HHMMSS.log
#   logs/twitter2015/none_seq1_YYYYMMDD_HHMMSS.log
#===============================================================================

# 设置错误时退出
set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 创建日志目录
LOG_DIR="logs/twitter2015"
mkdir -p "$LOG_DIR"

# 获取时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 定义配置文件和日志文件
DEQA_CONFIG="scripts/configs/server_twitter2015_deqa_seq1.json"
MOE_CONFIG="scripts/configs/server_twitter2015_moe_seq1.json"
NONE_CONFIG="scripts/configs/server_twitter2015_none_seq1.json"

DEQA_LOG="${LOG_DIR}/deqa_seq1_${TIMESTAMP}.log"
MOE_LOG="${LOG_DIR}/moe_seq1_${TIMESTAMP}.log"
NONE_LOG="${LOG_DIR}/none_seq1_${TIMESTAMP}.log"

# 定义PID文件
DEQA_PID="${LOG_DIR}/deqa_seq1.pid"
MOE_PID="${LOG_DIR}/moe_seq1.pid"
NONE_PID="${LOG_DIR}/none_seq1.pid"

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

# 显示GPU状态
show_gpu_status() {
    print_separator
    echo -e "${CYAN}当前GPU状态:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s: %s | 显存: %s/%s MB | 利用率: %s%%\n", $1, $2, $3, $4, $5}'
    else
        print_warning "nvidia-smi未找到"
    fi
    print_separator
}

# 检查Python环境
check_python() {
    if ! command -v python &> /dev/null; then
        print_error "Python not found! Please activate your conda/venv environment."
        exit 1
    fi
    
    PYTHON_VERSION=$(python --version 2>&1)
    print_info "Python环境: $PYTHON_VERSION"
}

# 检查配置文件
check_config() {
    local config_file=$1
    if [ ! -f "$config_file" ]; then
        print_error "配置文件不存在: $config_file"
        exit 1
    fi
    print_info "配置文件检查通过: $config_file"
}

# 等待GPU空闲
wait_for_gpu() {
    local gpu_id=$1
    local threshold=1000  # 显存使用阈值(MB)
    
    while true; do
        local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
        if [ "$mem_used" -lt "$threshold" ]; then
            break
        fi
        print_info "GPU $gpu_id 显存使用 ${mem_used}MB，等待空闲..."
        sleep 30
    done
}

# 运行单个实验（指定GPU）
run_experiment_on_gpu() {
    local strategy=$1
    local config=$2
    local log_file=$3
    local pid_file=$4
    local gpu_id=$5
    
    print_separator
    print_gpu_info "${strategy} 策略将运行在 GPU ${gpu_id}"
    print_info "配置文件: $config"
    print_info "日志文件: $log_file"
    print_separator
    
    # 设置CUDA_VISIBLE_DEVICES，只使用指定的GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # 使用nohup在后台运行
    nohup python -u -m scripts.train_with_zero_shot \
        --config "$config" \
        --start_task 0 \
        --end_task 8 \
        > "$log_file" 2>&1 &
    
    # 保存PID
    local pid=$!
    echo $pid > "$pid_file"
    
    print_info "${strategy} 已启动 (PID: $pid, GPU: $gpu_id)"
    print_info "查看实时日志: tail -f $log_file"
    print_info "停止任务: kill $pid"
    echo ""
    
    # 等待确认进程启动
    sleep 5
    if ps -p $pid > /dev/null; then
        print_info "${strategy} 运行正常 ✓"
    else
        print_error "${strategy} 启动失败，请查看日志: $log_file"
        exit 1
    fi
    
    # 取消CUDA_VISIBLE_DEVICES的导出
    unset CUDA_VISIBLE_DEVICES
}

# 监控所有实验进程
monitor_experiments() {
    print_separator
    echo -e "${CYAN}实验监控面板${NC}"
    print_separator
    
    local all_pids=()
    
    # 收集所有PID
    if [ -f "$DEQA_PID" ]; then
        all_pids+=($(cat "$DEQA_PID"))
    fi
    if [ -f "$MOE_PID" ]; then
        all_pids+=($(cat "$MOE_PID"))
    fi
    if [ -f "$NONE_PID" ]; then
        all_pids+=($(cat "$NONE_PID"))
    fi
    
    # 显示运行状态
    echo ""
    echo -e "${GREEN}运行中的实验:${NC}"
    for pid in "${all_pids[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            local cmd=$(ps -p $pid -o args= | grep -o 'server_twitter2015_[^/]*')
            echo "  ✓ PID $pid: $cmd"
        else
            echo "  ✗ PID $pid: 已结束"
        fi
    done
    
    echo ""
    echo -e "${GREEN}GPU使用情况:${NC}"
    show_gpu_status
}

# 创建监控脚本
create_monitor_script() {
    local monitor_script="${LOG_DIR}/monitor.sh"
    
    cat > "$monitor_script" << 'MONITOR_EOF'
#!/bin/bash
# 实验监控脚本
# 使用: bash logs/twitter2015/monitor.sh

LOG_DIR="logs/twitter2015"

while true; do
    clear
    echo "==============================================================================="
    echo "                     Twitter2015 实验监控面板"
    echo "                     $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==============================================================================="
    echo ""
    
    # 检查进程状态
    echo "运行中的实验:"
    ps aux | grep "train_with_zero_shot.*server_twitter2015" | grep -v grep | \
    awk '{print "  PID: " $2 " | CPU: " $3 "% | MEM: " $4 "% | " $NF}'
    
    echo ""
    echo "-------------------------------------------------------------------------------"
    echo "GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s\n    显存: %s/%s MB | 利用率: %s%% | 温度: %s°C\n", 
        $1, $2, $3, $4, $5, $6}'
    
    echo ""
    echo "-------------------------------------------------------------------------------"
    echo "最新日志 (最后5行):"
    echo ""
    
    for log in ${LOG_DIR}/deqa_seq1_*.log ${LOG_DIR}/moe_seq1_*.log ${LOG_DIR}/none_seq1_*.log; do
        if [ -f "$log" ]; then
            echo "$(basename $log):"
            tail -n 3 "$log" | sed 's/^/    /'
            echo ""
        fi
    done
    
    echo "==============================================================================="
    echo "按 Ctrl+C 退出监控"
    echo "==============================================================================="
    
    sleep 10
done
MONITOR_EOF
    
    chmod +x "$monitor_script"
    print_info "监控脚本已创建: $monitor_script"
}

#===============================================================================
# 主流程
#===============================================================================

print_separator
echo -e "${BLUE}Twitter2015 多策略对比实验 - 多GPU优化版${NC}"
echo -e "${BLUE}任务序列: masc → mate → mner → mabsa (x2轮)${NC}"
echo -e "${BLUE}模式序列: text_only (x4) → multimodal (x4)${NC}"
print_separator
echo ""

# 1. 检测GPU
print_info "Step 1: 检测GPU资源"
GPU_COUNT=$(detect_gpus)
print_gpu_info "检测到 ${GPU_COUNT} 个GPU"
show_gpu_status
echo ""

# 2. 检查环境
print_info "Step 2: 检查Python环境"
check_python
echo ""

# 3. 检查配置文件
print_info "Step 3: 检查配置文件"
check_config "$DEQA_CONFIG"
check_config "$MOE_CONFIG"
check_config "$NONE_CONFIG"
echo ""

# 4. 规划GPU分配策略
print_info "Step 4: 规划GPU分配策略"
echo ""

if [ "$GPU_COUNT" -ge 3 ]; then
    print_gpu_info "3+ GPU模式: 三个实验完全并行"
    echo "  GPU 0: DEQA"
    echo "  GPU 1: MoE-Adapters"
    echo "  GPU 2: None (微调)"
    STRATEGY_MODE="parallel_3"
elif [ "$GPU_COUNT" -eq 2 ]; then
    print_gpu_info "2 GPU模式: 前两个并行，第三个等待"
    echo "  GPU 0: DEQA"
    echo "  GPU 1: MoE-Adapters"
    echo "  等待: None (在前两个完成后运行)"
    STRATEGY_MODE="parallel_2"
else
    print_gpu_info "1 GPU模式: 三个实验串行运行"
    echo "  GPU 0: DEQA → MoE → None (依次运行)"
    STRATEGY_MODE="serial"
fi
echo ""

# 5. 用户确认
echo -e "${YELLOW}预计总时间:${NC}"
if [ "$STRATEGY_MODE" == "parallel_3" ]; then
    echo "  约 15-20小时 (三个实验完全并行)"
elif [ "$STRATEGY_MODE" == "parallel_2" ]; then
    echo "  约 30-35小时 (两个并行 + 一个串行)"
else
    echo "  约 45-60小时 (三个实验串行)"
fi
echo ""

read -p "是否继续？[y/N] " -r confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    print_info "已取消"
    exit 0
fi
echo ""

# 6. 运行实验
print_info "Step 5: 启动实验"
echo ""

if [ "$STRATEGY_MODE" == "parallel_3" ]; then
    # 三GPU完全并行
    run_experiment_on_gpu "DEQA" "$DEQA_CONFIG" "$DEQA_LOG" "$DEQA_PID" 0
    run_experiment_on_gpu "MoE-Adapters" "$MOE_CONFIG" "$MOE_LOG" "$MOE_PID" 1
    run_experiment_on_gpu "None/微调" "$NONE_CONFIG" "$NONE_LOG" "$NONE_PID" 2
    
elif [ "$STRATEGY_MODE" == "parallel_2" ]; then
    # 两GPU并行 + 一个等待
    run_experiment_on_gpu "DEQA" "$DEQA_CONFIG" "$DEQA_LOG" "$DEQA_PID" 0
    run_experiment_on_gpu "MoE-Adapters" "$MOE_CONFIG" "$MOE_LOG" "$MOE_PID" 1
    
    print_separator
    print_info "None实验将在DEQA或MoE完成后自动启动"
    print_info "创建后台监控任务..."
    print_separator
    
    # 创建后台任务，监控前两个实验，完成后启动第三个
    (
        # 等待DEQA或MoE任意一个完成
        while true; do
            deqa_running=0
            moe_running=0
            
            if [ -f "$DEQA_PID" ] && ps -p $(cat "$DEQA_PID") > /dev/null 2>&1; then
                deqa_running=1
            fi
            
            if [ -f "$MOE_PID" ] && ps -p $(cat "$MOE_PID") > /dev/null 2>&1; then
                moe_running=1
            fi
            
            # 如果任意一个完成，选择空闲的GPU
            if [ $deqa_running -eq 0 ]; then
                gpu_for_none=0
                break
            elif [ $moe_running -eq 0 ]; then
                gpu_for_none=1
                break
            fi
            
            sleep 60  # 每分钟检查一次
        done
        
        # 启动None实验
        print_info "检测到GPU ${gpu_for_none} 空闲，启动None实验"
        wait_for_gpu $gpu_for_none  # 等待GPU完全空闲
        run_experiment_on_gpu "None/微调" "$NONE_CONFIG" "$NONE_LOG" "$NONE_PID" $gpu_for_none
    ) > "${LOG_DIR}/scheduler_${TIMESTAMP}.log" 2>&1 &
    
    scheduler_pid=$!
    echo $scheduler_pid > "${LOG_DIR}/scheduler.pid"
    print_info "调度器已启动 (PID: $scheduler_pid)"
    
else
    # 单GPU串行
    run_experiment_on_gpu "DEQA" "$DEQA_CONFIG" "$DEQA_LOG" "$DEQA_PID" 0
    
    # 等待DEQA完成
    deqa_pid=$(cat "$DEQA_PID")
    print_info "等待DEQA完成..."
    wait $deqa_pid
    
    run_experiment_on_gpu "MoE-Adapters" "$MOE_CONFIG" "$MOE_LOG" "$MOE_PID" 0
    
    # 等待MoE完成
    moe_pid=$(cat "$MOE_PID")
    print_info "等待MoE完成..."
    wait $moe_pid
    
    run_experiment_on_gpu "None/微调" "$NONE_CONFIG" "$NONE_LOG" "$NONE_PID" 0
fi

# 7. 创建监控工具
create_monitor_script

# 8. 显示总结
print_separator
print_info "所有实验已启动！"
print_separator
echo ""

# 显示日志查看命令
echo -e "${GREEN}📊 查看实时日志:${NC}"
echo "  DEQA:   tail -f $DEQA_LOG"
echo "  MoE:    tail -f $MOE_LOG"
if [ "$STRATEGY_MODE" == "parallel_3" ]; then
    echo "  None:   tail -f $NONE_LOG"
elif [ "$STRATEGY_MODE" == "parallel_2" ]; then
    echo "  None:   (等待中，将在前两个实验之一完成后启动)"
fi
echo ""

# 显示监控命令
echo -e "${GREEN}📈 实时监控面板:${NC}"
echo "  bash ${LOG_DIR}/monitor.sh"
echo ""

# 显示GPU监控
echo -e "${GREEN}🖥️  GPU使用监控:${NC}"
echo "  watch -n 1 nvidia-smi"
echo ""

# 显示进程管理
echo -e "${GREEN}🔧 进程管理:${NC}"
echo "  查看状态: ps aux | grep train_with_zero_shot"
echo "  停止DEQA: kill \$(cat $DEQA_PID)"
echo "  停止MoE:  kill \$(cat $MOE_PID)"
if [ "$STRATEGY_MODE" != "serial" ]; then
    echo "  停止None: kill \$(cat $NONE_PID)"
fi
echo ""

print_separator
print_info "✅ 实验在后台运行，SSH断开后会继续执行"
print_info "✅ 所有输出已重定向到日志文件"
print_info "✅ GPU资源已优化分配"
print_separator

