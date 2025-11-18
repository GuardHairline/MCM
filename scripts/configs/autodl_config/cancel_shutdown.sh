#!/bin/bash
# 取消AutoDL自动关机
#
# 如果需要在实验完成后继续使用服务器，运行此脚本
#
# 使用方法：
#   bash scripts/configs/autodl_config/cancel_shutdown.sh

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}========================================================================"
echo -e "取消自动关机"
echo -e "========================================================================${NC}"
echo ""

# 检查是否有计划的关机
if sudo shutdown -c 2>/dev/null; then
    echo -e "${GREEN}[SUCCESS]${NC} 已取消计划的关机"
else
    echo -e "${YELLOW}[INFO]${NC} 没有找到计划的关机任务"
fi

echo ""
echo -e "${YELLOW}[WARNING]${NC} 注意："
echo "  - AutoDL按时计费，继续运行会产生费用"
echo "  - 完成工作后记得手动关机"
echo "  - 手动关机命令: sudo shutdown -h now"
echo ""







