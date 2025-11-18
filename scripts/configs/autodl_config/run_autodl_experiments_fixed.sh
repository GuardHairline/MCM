#!/bin/bash
# AutoDL服务器实验运行脚本（修复跨日期问题）
# 
# 特点：
# 1. 独占GPU，无需等待
# 2. 依次运行所有实验
# 3. 记录详细的成功/失败信息
# 4. 发送邮件通知实验结果
# 5. 完成后自动关机（节省费用）
# 6. 【新增】自动检测配置文件中的日期，避免跨日期问题
#
# 使用方法：
#   bash scripts/configs/autodl_config/run_autodl_experiments_fixed.sh --email your@email.com
#
# 或指定日期文件夹：
#   bash scripts/configs/autodl_config/run_autodl_experiments_fixed.sh --date 251022 --email your@email.com
#
# 注意：
# - 确保已上传所有数据文件
# - 确保环境已配置（conda activate）
# - 建议使用nohup运行并退出SSH
# - 如果配置文件和运行不在同一天，脚本会自动检测并警告

# ============================================================================
# 配置
# ============================================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 日期文件夹（可通过命令行参数覆盖）
DATE_FOLDER=""
AUTO_DETECT_DATE=true

# GPU设备
export CUDA_VISIBLE_DEVICES=0

# 邮件地址（从命令行参数获取）
EMAIL_ADDRESS=""

# ============================================================================
# 解析命令行参数
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --date)
            DATE_FOLDER="$2"
            AUTO_DETECT_DATE=false
            shift 2
            ;;
        --email)
            EMAIL_ADDRESS="$2"
            shift 2
            ;;
        --smtp-user)
            SMTP_USER="$2"
            shift 2
            ;;
        --smtp-password)
            SMTP_PASSWORD="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "使用方法: $0 [--date YYMMDD] --email your@email.com [--smtp-user user] [--smtp-password pass]"
            exit 1
            ;;
    esac
done

# ============================================================================
# 智能日期检测
# ============================================================================

# 任务列表
CONFIG_INDEX="$SCRIPT_DIR/config_index.json"

if [ "$AUTO_DETECT_DATE" = true ]; then
    echo "🔍 自动检测配置文件中的日期..."
    
    # 检查配置索引是否存在
    if [ ! -f "$CONFIG_INDEX" ]; then
        echo "❌ 配置索引不存在: $CONFIG_INDEX"
        echo "请先运行: python scripts/generate_autodl_configs.py"
        exit 1
    fi
    
    # 从第一个配置文件中提取日期
    first_config=$(python3 -c "
import json
with open('$CONFIG_INDEX', 'r') as f:
    configs = json.load(f)['configs']
    if configs:
        print(configs[0]['path'])
    else:
        print('')
")
    
    if [ -z "$first_config" ]; then
        echo "❌ 配置索引中没有找到配置文件"
        exit 1
    fi
    
    # 从配置文件中提取日期文件夹
    # 示例路径: scripts/configs/autodl_config/autodl_seq1_twitter2015_deqa.json
    # 配置内容包含: /root/autodl-tmp/checkpoints/251022/...
    DETECTED_DATE=$(python3 -c "
import json
with open('$first_config', 'r') as f:
    config = json.load(f)
    checkpoint_dir = config['global_params']['checkpoint_dir']
    # 从路径中提取日期: /root/autodl-tmp/checkpoints/251022 -> 251022
    import os
    date_folder = os.path.basename(checkpoint_dir)
    print(date_folder)
")
    
    if [ -z "$DETECTED_DATE" ]; then
        echo "⚠️  无法从配置文件中检测日期，使用当前日期"
        DATE_FOLDER=$(date +%y%m%d)
    else
        DATE_FOLDER="$DETECTED_DATE"
        echo "✅ 检测到配置文件日期: $DATE_FOLDER"
        
        # 检查是否与当前日期一致
        CURRENT_DATE=$(date +%y%m%d)
        if [ "$DATE_FOLDER" != "$CURRENT_DATE" ]; then
            echo ""
            echo "⚠️  ============================================================"
            echo "⚠️  警告：日期不一致！"
            echo "⚠️  ============================================================"
            echo "⚠️  配置文件日期: $DATE_FOLDER"
            echo "⚠️  当前系统日期: $CURRENT_DATE"
            echo "⚠️  "
            echo "⚠️  这可能是因为："
            echo "⚠️  1. 配置文件是之前生成的"
            echo "⚠️  2. 系统时间有误"
            echo "⚠️  "
            echo "⚠️  脚本将使用配置文件中的日期: $DATE_FOLDER"
            echo "⚠️  如需重新生成配置，请运行:"
            echo "⚠️    python scripts/generate_autodl_configs.py"
            echo "⚠️  ============================================================"
            echo ""
            
            # 给用户10秒时间取消
            echo "⏳ 10秒后继续执行... (Ctrl+C 取消)"
            for i in {10..1}; do
                echo -ne "\r   倒计时: $i 秒..."
                sleep 1
            done
            echo ""
            echo ""
        fi
    fi
fi

# AutoDL数据盘路径
AUTODL_STORAGE="/root/autodl-tmp/checkpoints/$DATE_FOLDER"
LOG_DIR="$AUTODL_STORAGE/log"

# 创建目录
mkdir -p "$LOG_DIR"

# 运行日志
RUN_LOG="$LOG_DIR/autodl_run_$(date +%Y%m%d_%H%M%S).log"
PROGRESS_LOG="$LOG_DIR/autodl_progress.json"
RESULT_JSON="$LOG_DIR/autodl_result.json"

# ============================================================================
# 工具函数
# ============================================================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$RUN_LOG"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$RUN_LOG"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$RUN_LOG"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$RUN_LOG"
}

# 记录进度
update_progress() {
    local total=$1
    local completed=$2
    local failed=$3
    local current_config=$4
    
    cat > "$PROGRESS_LOG" << EOF
{
    "total": $total,
    "completed": $completed,
    "failed": $failed,
    "remaining": $((total - completed - failed)),
    "current": "$current_config",
    "last_update": "$(date '+%Y-%m-%d %H:%M:%S')",
    "date_folder": "$DATE_FOLDER"
}
EOF
}

# 保存最终结果
save_final_result() {
    local total=$1
    local completed=$2
    local failed=$3
    local start_time=$4
    local end_time=$5
    local duration=$6
    
    cat > "$RESULT_JSON" << EOF
{
    "total": $total,
    "completed": $completed,
    "failed": $failed,
    "start_time": "$start_time",
    "end_time": "$end_time",
    "duration_seconds": $duration,
    "date_folder": "$DATE_FOLDER",
    "storage_path": "$AUTODL_STORAGE",
    "successful_configs": [
EOF
    
    # 添加成功的配置
    local first=true
    for ((i=0; i<${#successful_configs[@]}; i++)); do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$RESULT_JSON"
        fi
        echo "        {" >> "$RESULT_JSON"
        echo "            \"name\": \"${successful_configs[$i]}\"," >> "$RESULT_JSON"
        echo "            \"duration\": ${successful_durations[$i]}" >> "$RESULT_JSON"
        echo -n "        }" >> "$RESULT_JSON"
    done
    
    cat >> "$RESULT_JSON" << EOF

    ],
    "failed_configs": [
EOF
    
    # 添加失败的配置
    first=true
    for ((i=0; i<${#failed_configs[@]}; i++)); do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$RESULT_JSON"
        fi
        # 转义错误消息中的引号和换行
        local error_msg="${failed_errors[$i]}"
        error_msg="${error_msg//\\/\\\\}"
        error_msg="${error_msg//\"/\\\"}"
        error_msg="${error_msg//$'\n'/\\n}"
        
        echo "        {" >> "$RESULT_JSON"
        echo "            \"name\": \"${failed_configs[$i]}\"," >> "$RESULT_JSON"
        echo "            \"error\": \"$error_msg\"" >> "$RESULT_JSON"
        echo -n "        }" >> "$RESULT_JSON"
    done
    
    cat >> "$RESULT_JSON" << EOF

    ]
}
EOF
    
    print_info "结果已保存到: $RESULT_JSON"
}

# 发送邮件通知
send_email_notification() {
    if [ -z "$EMAIL_ADDRESS" ]; then
        print_warning "未配置邮件地址，跳过邮件通知"
        return 0
    fi
    
    print_info "发送邮件通知到: $EMAIL_ADDRESS"
    
    cd "$PROJECT_ROOT"
    
    local email_cmd="python scripts/configs/autodl_config/send_email_notification.py --email $EMAIL_ADDRESS --result $RESULT_JSON"
    
    if [ -n "$SMTP_USER" ]; then
        email_cmd="$email_cmd --smtp-user $SMTP_USER"
    fi
    
    if [ -n "$SMTP_PASSWORD" ]; then
        email_cmd="$email_cmd --smtp-password $SMTP_PASSWORD"
    fi
    
    if eval "$email_cmd"; then
        print_success "邮件发送成功"
    else
        print_warning "邮件发送失败，请检查SMTP配置"
    fi
}

# 关机
shutdown_server() {
    print_warning "准备关机以节省费用..."
    print_warning "10秒后将执行关机命令"
    
    # 给予10秒缓冲时间
    for i in {10..1}; do
        echo -ne "\r关机倒计时: $i 秒... (Ctrl+C 取消)"
        sleep 1
    done
    echo ""
    
    print_success "所有任务已完成，正在关机..."
    
    # AutoDL使用标准关机命令
    sudo shutdown -h now
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    local start_timestamp=$(date +%s)
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    print_info "========================================================================"
    print_info "AutoDL服务器 - 自动实验运行（修复版）"
    print_info "========================================================================"
    print_info ""
    print_info "开始时间: $start_time"
    print_info "项目目录: $PROJECT_ROOT"
    print_info "日期文件夹: $DATE_FOLDER"
    print_info "存储目录: $AUTODL_STORAGE"
    print_info "日志文件: $RUN_LOG"
    
    if [ -n "$EMAIL_ADDRESS" ]; then
        print_info "邮件通知: $EMAIL_ADDRESS"
    else
        print_warning "邮件通知: 未配置（使用 --email 参数配置）"
    fi
    
    print_info ""
    
    # 切换到项目目录
    cd "$PROJECT_ROOT" || exit 1
    
    # 检查配置索引
    if [ ! -f "$CONFIG_INDEX" ]; then
        print_error "配置索引不存在: $CONFIG_INDEX"
        print_error "请先运行: python scripts/generate_autodl_configs.py"
        exit 1
    fi
    
    # 检查GPU
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi 未找到，无法使用GPU"
        exit 1
    fi
    
    print_info "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$RUN_LOG"
    print_info ""
    
    # 获取配置列表
    total_configs=$(python3 -c "import json; print(json.load(open('$CONFIG_INDEX'))['total'])")
    print_info "总实验数: $total_configs"
    print_info "执行顺序: seq1->seq2, twitter2015->twitter2017->mix, deqa->none->moe->..."
    print_info ""
    
    # 计数器和记录数组
    local completed=0
    local failed=0
    declare -a successful_configs
    declare -a successful_durations
    declare -a failed_configs
    declare -a failed_errors
    
    # 读取配置并运行
    python3 -c "
import json
with open('$CONFIG_INDEX', 'r') as f:
    configs = json.load(f)['configs']
    for cfg in configs:
        print(cfg['path'])
" | while read config_path; do
        
        config_name=$(basename "$config_path")
        
        print_info "========================================================================"
        print_info "运行实验: $config_name ($((completed + failed + 1))/$total_configs)"
        print_info "========================================================================"
        
        # 更新进度
        update_progress "$total_configs" "$completed" "$failed" "$config_name"
        
        # 运行实验
        task_start_time=$(date +%s)
        
        print_info "执行命令: python -m scripts.train_with_zero_shot --config $config_path --start_task 0 --end_task 8"
        
        # 捕获输出和错误
        task_log="$LOG_DIR/${config_name%.json}.log"
        
        if python -m scripts.train_with_zero_shot \
            --config "$config_path" \
            --start_task 0 \
            --end_task 8 \
            > "$task_log" 2>&1; then
            
            task_end_time=$(date +%s)
            task_duration=$((task_end_time - task_start_time))
            
            print_success "实验完成: $config_name (耗时: ${task_duration}s)"
            successful_configs+=("$config_name")
            successful_durations+=("$task_duration")
            completed=$((completed + 1))
            
        else
            task_end_time=$(date +%s)
            task_duration=$((task_end_time - task_start_time))
            
            # 提取错误信息
            error_msg=$(tail -20 "$task_log" | grep -E "Error|Exception|Traceback" | head -5 | tr '\n' ' ')
            if [ -z "$error_msg" ]; then
                error_msg="Unknown error - check log: $task_log"
            fi
            
            print_error "实验失败: $config_name (耗时: ${task_duration}s)"
            print_error "错误: $error_msg"
            
            failed_configs+=("$config_name")
            failed_errors+=("$error_msg")
            failed=$((failed + 1))
            
            # 失败后继续
            print_warning "继续运行下一个实验..."
        fi
        
        print_info ""
        print_info "当前进度: 完成=$completed, 失败=$failed, 剩余=$((total_configs - completed - failed))"
        print_info ""
        
        # 短暂休息
        sleep 3
    done
    
    # 获取最终时间
    local end_timestamp=$(date +%s)
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    local total_duration=$((end_timestamp - start_timestamp))
    
    # 保存最终结果
    save_final_result "$total_configs" "$completed" "$failed" \
        "$start_time" "$end_time" "$total_duration"
    
    # 最终统计
    print_info "========================================================================"
    print_info "所有实验运行完成"
    print_info "========================================================================"
    print_info ""
    print_info "结束时间: $end_time"
    print_info "总实验数: $total_configs"
    print_success "成功: $completed"
    
    if [ $failed -gt 0 ]; then
        print_error "失败: $failed"
        print_error "失败的实验:"
        for cfg in "${failed_configs[@]}"; do
            print_error "  - $cfg"
        done
    fi
    
    print_info ""
    print_info "详细结果: $RESULT_JSON"
    print_info "日志目录: $LOG_DIR"
    print_info "存储路径: $AUTODL_STORAGE"
    print_info ""
    
    # 发送邮件通知
    send_email_notification
    
    # 检查是否需要关机
    print_warning "========================================================================"
    print_warning "实验已完成，准备关机以节省费用"
    print_warning "========================================================================"
    print_warning ""
    print_warning "如果不想关机，请在10秒内按 Ctrl+C"
    print_warning ""
    
    # 执行关机
    shutdown_server
}

# 捕获中断信号
trap 'print_warning "收到中断信号，停止运行"; exit 1' INT TERM

# 运行主流程
main

# 如果到这里说明关机失败
print_error "关机命令执行失败，请手动关机"
print_error "命令: sudo shutdown -h now"

