#!/bin/bash
# AutoDL实验停止脚本
#
# 紧急停止正在运行的实验
#
# 使用方法：
#   bash scripts/configs/autodl_config/stop_experiments.sh

# 颜色
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${YELLOW}========================================================================"
echo -e "停止AutoDL实验"
echo -e "========================================================================${NC}"
echo ""

# 查找运行中的Python进程
echo -e "${YELLOW}[SEARCH]${NC} 查找运行中的实验进程..."
echo ""

pids=$(ps aux | grep "scripts.train_with_zero_shot" | grep -v grep | awk '{print $2}')

if [ -z "$pids" ]; then
    echo -e "${GREEN}[INFO]${NC} 没有找到运行中的实验"
    exit 0
fi

echo "找到以下进程:"
ps aux | grep "scripts.train_with_zero_shot" | grep -v grep
echo ""

# 确认
echo -e "${YELLOW}[WARNING]${NC} 确认要停止这些进程吗? (y/N)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${YELLOW}[STOP]${NC} 正在停止进程..."
    
    for pid in $pids; do
        echo "  停止进程 $pid"
        kill -TERM $pid 2>/dev/null
    done
    
    # 等待5秒
    echo ""
    echo -e "${YELLOW}[WAIT]${NC} 等待进程正常退出..."
    sleep 5
    
    # 检查是否还有进程
    remaining=$(ps aux | grep "scripts.train_with_zero_shot" | grep -v grep | awk '{print $2}')
    
    if [ -n "$remaining" ]; then
        echo -e "${YELLOW}[FORCE]${NC} 强制终止残留进程..."
        for pid in $remaining; do
            echo "  强制终止进程 $pid"
            kill -9 $pid 2>/dev/null
        done
    fi
    
    echo ""
    echo -e "${GREEN}[SUCCESS]${NC} 所有实验进程已停止"
    echo ""
    echo "注意："
    echo "  - 实验数据可能未保存完整"
    echo "  - 可以重新运行实验继续训练"
    echo "  - AutoDL服务器仍在运行，记得手动关机"
    echo ""
else
    echo ""
    echo -e "${GREEN}[CANCEL]${NC} 取消停止操作"
    echo ""
fi

