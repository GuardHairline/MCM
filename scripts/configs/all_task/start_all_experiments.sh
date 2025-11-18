#!/bin/bash
#===============================================================================
# 全任务序列实验 - 后台启动脚本
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/run_all_experiments.sh"
LOG_DIR="scripts/configs/all_task/logs"
MASTER_LOG="scripts/configs/all_task/logs/master_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}========================================================================${NC}"
echo -e "${GREEN}全任务序列实验 - 后台启动${NC}"
echo -e "${GREEN}========================================================================${NC}"
echo ""

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo -e "${YELLOW}[ERROR]${NC} 主脚本不存在: $MAIN_SCRIPT"
    exit 1
fi

echo -e "${CYAN}[INFO]${NC} 启动主实验脚本..."
echo -e "${CYAN}[INFO]${NC} 主日志文件: $MASTER_LOG"
echo ""

nohup bash "$MAIN_SCRIPT" > "$MASTER_LOG" 2>&1 &
MASTER_PID=$!

echo $MASTER_PID > "${LOG_DIR}/master.pid"

echo -e "${GREEN}✓ 主脚本已在后台启动${NC}"
echo -e "${GREEN}✓ 主进程PID: $MASTER_PID${NC}"
echo ""

sleep 3

if ps -p $MASTER_PID > /dev/null; then
    echo -e "${GREEN}✓ 实验已成功启动并在后台运行${NC}"
    echo ""
    echo -e "${CYAN}监控命令:${NC}"
    echo -e "  tail -f $MASTER_LOG"
    echo -e "  watch -n 1 nvidia-smi"
    echo ""
    echo -e "${CYAN}停止命令:${NC}"
    echo -e "  bash $SCRIPT_DIR/stop_all_experiments.sh"
    echo ""
    echo -e "${GREEN}现在可以安全地断开SSH连接${NC}"
else
    echo -e "${YELLOW}[ERROR]${NC} 主脚本启动失败，请检查日志: $MASTER_LOG"
    exit 1
fi
