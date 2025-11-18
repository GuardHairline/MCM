#!/bin/bash
# 设置AutoDL脚本执行权限
#
# 在AutoDL服务器上运行一次即可
#
# 使用方法：
#   bash scripts/configs/autodl_config/setup_permissions.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "设置AutoDL脚本执行权限..."
echo ""

# 添加执行权限
chmod +x "$SCRIPT_DIR"/run_autodl_experiments.sh
chmod +x "$SCRIPT_DIR"/test_single_autodl.sh
chmod +x "$SCRIPT_DIR"/check_progress.sh
chmod +x "$SCRIPT_DIR"/stop_experiments.sh
chmod +x "$SCRIPT_DIR"/cancel_shutdown.sh
chmod +x "$SCRIPT_DIR"/setup_permissions.sh

echo "✓ run_autodl_experiments.sh"
echo "✓ test_single_autodl.sh"
echo "✓ check_progress.sh"
echo "✓ stop_experiments.sh"
echo "✓ cancel_shutdown.sh"
echo "✓ setup_permissions.sh"
echo ""
echo "权限设置完成！"
echo ""
echo "下一步："
echo "  bash scripts/configs/autodl_config/test_single_autodl.sh"
echo ""







