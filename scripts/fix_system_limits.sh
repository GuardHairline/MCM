#!/bin/bash

# 系统资源检查和修复脚本
# 解决"Too many open files"等问题

echo "=== 系统资源检查和修复 ==="

# 检查当前文件描述符限制
echo "当前文件描述符限制:"
ulimit -n

# 检查系统级别的文件描述符限制
echo ""
echo "系统级别文件描述符限制:"
cat /proc/sys/fs/file-max 2>/dev/null || echo "无法读取系统级别限制"

# 尝试增加文件描述符限制
echo ""
echo "尝试增加文件描述符限制..."
ulimit -n 65536

# 验证是否成功
new_limit=$(ulimit -n)
echo "新的文件描述符限制: $new_limit"

if [ "$new_limit" -ge 65536 ]; then
    echo "✓ 文件描述符限制设置成功"
else
    echo "✗ 文件描述符限制设置失败，可能需要root权限"
    echo "请尝试以下命令："
    echo "  sudo ulimit -n 65536"
    echo "  或者在 /etc/security/limits.conf 中添加："
    echo "  * soft nofile 65536"
    echo "  * hard nofile 65536"
fi

# 检查内存使用情况
echo ""
echo "=== 内存使用情况 ==="
free -h

# 检查磁盘使用情况
echo ""
echo "=== 磁盘使用情况 ==="
df -h .

# 检查GPU使用情况（如果可用）
echo ""
echo "=== GPU使用情况 ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
else
    echo "GPU信息不可用"
fi

# 检查Python环境
echo ""
echo "=== Python环境信息 ==="
python --version
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch未安装')"

echo ""
echo "=== 检查完成 ===" 