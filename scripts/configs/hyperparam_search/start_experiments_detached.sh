#!/bin/bash
#===============================================================================
# MASC 超参数搜索 - 后台启动脚本
# 
# 此脚本可以完全脱离SSH运行，即使SSH断开连接也会继续执行所有实验
#
# 使用方法:
#   bash start_experiments_detached.sh
#
# 检查运行状态:
#   tail -f scripts/configs/hyperparam_search/logs/master_*.log
#   ps aux | grep train_with_zero_shot
#
# 停止所有实验:
#   bash scripts/configs/hyperparam_search/stop_all_experiments.sh
#===============================================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/run_all_experiments.sh"
LOG_DIR="scripts/configs/hyperparam_search/logs"
MASTER_LOG="scripts/configs/hyperparam_search/logs/master_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 打印路径信息便于调试
echo "脚本目录: $SCRIPT_DIR"
echo "主脚本: $MAIN_SCRIPT"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}========================================================================${NC}"
echo -e "${GREEN}MASC 超参数搜索实验 - 后台启动${NC}"
echo -e "${GREEN}========================================================================${NC}"
echo ""

# 检查主脚本是否存在
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo -e "${YELLOW}[ERROR]${NC} 主脚本不存在: $MAIN_SCRIPT"
    exit 1
fi

# 使用nohup启动主脚本，完全脱离当前shell
echo -e "${CYAN}[INFO]${NC} 启动主实验脚本..."
echo -e "${CYAN}[INFO]${NC} 主日志文件: $MASTER_LOG"
echo ""

nohup bash "$MAIN_SCRIPT" > "$MASTER_LOG" 2>&1 &
MASTER_PID=$!

# 保存主进程PID
echo $MASTER_PID > "${LOG_DIR}/master.pid"

echo -e "${GREEN}✓ 主脚本已在后台启动${NC}"
echo -e "${GREEN}✓ 主进程PID: $MASTER_PID${NC}"
echo -e "${GREEN}✓ PID文件: ${LOG_DIR}/master.pid${NC}"
echo ""

# 等待一下确保进程启动
sleep 3

# 检查进程是否还在运行
if ps -p $MASTER_PID > /dev/null; then
    echo -e "${GREEN}✓ 实验已成功启动并在后台运行${NC}"
    echo ""
    echo -e "${CYAN}========================================================================${NC}"
    echo -e "${CYAN}监控命令:${NC}"
    echo -e "${CYAN}========================================================================${NC}"
    echo -e "  查看主日志:   tail -f $MASTER_LOG"
    echo -e "  查看实验日志: ls -lth $LOG_DIR/exp_*.log"
    echo -e "  检查进程:     ps aux | grep train_with_zero_shot"
    echo -e "  查看GPU使用:  watch -n 1 nvidia-smi"
    echo ""
    echo -e "${CYAN}停止命令:${NC}"
    echo -e "  停止所有实验: bash $SCRIPT_DIR/stop_all_experiments.sh"
    echo ""
    echo -e "${GREEN}========================================================================${NC}"
    echo -e "${GREEN}现在可以安全地断开SSH连接，实验将继续运行${NC}"
    echo -e "${GREEN}========================================================================${NC}"
else
    echo -e "${YELLOW}[ERROR]${NC} 主脚本启动失败，请检查日志: $MASTER_LOG"
    exit 1
fi
