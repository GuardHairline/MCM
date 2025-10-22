#!/bin/bash
# AutoDL单任务测试脚本
#
# 在运行全部实验前，先测试单个配置是否正常
#
# 使用方法：
#   bash scripts/configs/autodl_config/test_single_autodl.sh [config_name]
#
# 示例：
#   bash scripts/configs/autodl_config/test_single_autodl.sh autodl_twitter2015_deqa_seq1.json

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 配置文件
if [ -z "$1" ]; then
    # 默认测试第一个配置
    CONFIG_FILE="$SCRIPT_DIR/autodl_twitter2015_deqa_seq1.json"
else
    CONFIG_FILE="$SCRIPT_DIR/$1"
fi

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}[ERROR]${NC} 配置文件不存在: $CONFIG_FILE"
    echo ""
    echo "可用的配置文件:"
    ls "$SCRIPT_DIR"/*.json 2>/dev/null | head -5
    echo "..."
    exit 1
fi

echo -e "${BLUE}========================================================================"
echo -e "AutoDL单任务测试"
echo -e "========================================================================${NC}"
echo ""
echo -e "${BLUE}[INFO]${NC} 配置文件: $CONFIG_FILE"
echo -e "${BLUE}[INFO]${NC} 项目目录: $PROJECT_ROOT"
echo ""

# 切换目录
cd "$PROJECT_ROOT" || exit 1

# 显示GPU信息
echo -e "${BLUE}[INFO]${NC} GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""

# 运行测试
echo -e "${YELLOW}[TEST]${NC} 开始运行测试..."
echo -e "${YELLOW}[TEST]${NC} 只运行第一个任务以快速验证"
echo ""

start_time=$(date +%s)

# 只运行第一个任务
python -m scripts.train_with_zero_shot \
    --config "$CONFIG_FILE" \
    --start_task 0 \
    --end_task 1

exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo -e "${BLUE}========================================================================"
echo -e "测试结果"
echo -e "========================================================================${NC}"
echo ""

if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS]${NC} 测试通过！"
    echo -e "${GREEN}[SUCCESS]${NC} 耗时: ${duration}秒"
    echo ""
    echo "下一步："
    echo "  运行完整实验: bash scripts/configs/autodl_config/run_autodl_experiments.sh"
    echo ""
    exit 0
else
    echo -e "${RED}[FAILED]${NC} 测试失败"
    echo -e "${RED}[FAILED]${NC} 退出码: $exit_code"
    echo ""
    echo "请检查："
    echo "  1. 数据文件是否完整"
    echo "  2. 环境依赖是否安装"
    echo "  3. GPU是否可用"
    echo "  4. 配置文件是否正确"
    echo ""
    exit 1
fi

