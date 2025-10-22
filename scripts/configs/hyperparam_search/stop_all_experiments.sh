#!/bin/bash
#===============================================================================
# 停止所有MASC超参数搜索实验
#===============================================================================

LOG_DIR="scripts/configs/hyperparam_search/logs"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${RED}========================================================================${NC}"
echo -e "${RED}停止所有实验${NC}"
echo -e "${RED}========================================================================${NC}"
echo ""

# 停止主进程
if [ -f "${LOG_DIR}/master.pid" ]; then
    MASTER_PID=$(cat "${LOG_DIR}/master.pid")
    if ps -p $MASTER_PID > /dev/null; then
        echo -e "${YELLOW}停止主进程 (PID: $MASTER_PID)...${NC}"
        kill $MASTER_PID
    fi
    rm -f "${LOG_DIR}/master.pid"
fi

# 停止所有实验进程
for pid_file in "${LOG_DIR}"/*.pid; do
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null; then
            echo -e "${YELLOW}停止实验进程 (PID: $PID)...${NC}"
            kill $PID
        fi
        rm -f "$pid_file"
    fi
done

# 确保所有Python训练进程都被停止
echo ""
echo -e "${YELLOW}检查是否还有训练进程在运行...${NC}"
TRAIN_PIDS=$(ps aux | grep "train_with_zero_shot" | grep -v grep | awk '{print $2}')

if [ -n "$TRAIN_PIDS" ]; then
    echo -e "${YELLOW}发现残留进程，强制终止...${NC}"
    echo "$TRAIN_PIDS" | xargs kill -9 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}✓ 所有实验已停止${NC}"
echo ""
