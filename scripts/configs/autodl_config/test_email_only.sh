#!/bin/bash
# 仅测试邮件发送功能
# 使用方法: bash test_email_only.sh your@email.com

EMAIL="$1"

if [ -z "$EMAIL" ]; then
    echo "请提供邮箱地址"
    echo "使用方法: bash test_email_only.sh your@email.com"
    exit 1
fi

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 创建测试结果文件
TEST_RESULT="$SCRIPT_DIR/test_result.json"

cat > "$TEST_RESULT" << 'EOF'
{
    "total": 1,
    "completed": 1,
    "failed": 0,
    "start_time": "2024-10-22 10:00:00",
    "end_time": "2024-10-22 12:00:00",
    "duration_seconds": 7200,
    "successful_configs": [
        {
            "name": "server_twitter2015_deqa_seq1.json",
            "duration": 7200
        }
    ],
    "failed_configs": []
}
EOF

echo "测试邮件发送到: $EMAIL"
echo ""

# 发送测试邮件
python "$SCRIPT_DIR/send_email_notification.py" \
    --email "$EMAIL" \
    --result "$TEST_RESULT"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 邮件测试成功！请检查邮箱"
    echo ""
    echo "如果收到邮件，可以运行完整实验："
    echo "  bash scripts/configs/autodl_config/run_autodl_experiments.sh --email $EMAIL"
else
    echo ""
    echo "✗ 邮件测试失败"
    echo ""
    echo "请检查："
    echo "  1. 邮箱配置是否正确（send_email_notification.py）"
    echo "  2. 网络是否正常"
    echo "  3. 授权码是否有效"
fi

# 清理测试文件
rm -f "$TEST_RESULT"







