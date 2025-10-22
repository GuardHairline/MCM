#!/bin/bash
# AutoDL实验进度查看脚本
#
# 使用方法：
#   bash scripts/configs/autodl_config/check_progress.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PROGRESS_FILE="$PROJECT_ROOT/checkpoints/autodl/log/autodl_progress.json"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

clear

echo -e "${BLUE}========================================================================"
echo -e "AutoDL实验进度"
echo -e "========================================================================${NC}"
echo ""

# 检查进度文件
if [ ! -f "$PROGRESS_FILE" ]; then
    echo -e "${YELLOW}[WARNING]${NC} 进度文件不存在，实验可能未开始"
    echo ""
    echo "进度文件: $PROGRESS_FILE"
    exit 1
fi

# 显示进度
echo -e "${BLUE}[INFO]${NC} 当前状态:"
echo ""

python3 << 'EOF'
import json
import sys
from pathlib import Path

progress_file = Path(sys.argv[1])

try:
    with open(progress_file, 'r') as f:
        data = json.load(f)
    
    total = data.get('total', 0)
    completed = data.get('completed', 0)
    failed = data.get('failed', 0)
    remaining = data.get('remaining', 0)
    current = data.get('current', 'N/A')
    last_update = data.get('last_update', 'N/A')
    
    # 计算百分比
    progress_pct = (completed + failed) / total * 100 if total > 0 else 0
    
    print(f"  总任务数:   {total}")
    print(f"  已完成:     {completed} ({completed/total*100:.1f}%)" if total > 0 else f"  已完成:     {completed}")
    print(f"  已失败:     {failed} ({failed/total*100:.1f}%)" if total > 0 else f"  已失败:     {failed}")
    print(f"  剩余:       {remaining}")
    print(f"  总进度:     {progress_pct:.1f}%")
    print(f"")
    print(f"  当前任务:   {current}")
    print(f"  最后更新:   {last_update}")
    print()
    
    # 进度条
    bar_length = 50
    filled = int(bar_length * progress_pct / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"  [{bar}] {progress_pct:.1f}%")

except Exception as e:
    print(f"读取进度文件失败: {e}")
    sys.exit(1)

EOF "$PROGRESS_FILE"

echo ""
echo -e "${BLUE}========================================================================"
echo ""

# 显示最近的日志
echo -e "${BLUE}[INFO]${NC} 最近的日志 (最后20行):"
echo ""

latest_log=$(ls -t "$PROJECT_ROOT/checkpoints/autodl/log"/autodl_run_*.log 2>/dev/null | head -1)

if [ -n "$latest_log" ]; then
    tail -20 "$latest_log"
else
    echo "  没有找到日志文件"
fi

echo ""
echo -e "${BLUE}========================================================================"
echo ""
echo "刷新进度: bash $(basename $0)"
echo "查看完整日志: tail -f $PROJECT_ROOT/checkpoints/autodl/log/autodl_run_*.log"
echo "GPU状态: nvidia-smi"
echo ""

