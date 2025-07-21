#!/bin/bash

# 清理特定配置生成的模型文件
# 用法: ./scripts/cleanup_model_files.sh <config_name>
# 例如: ./scripts/cleanup_model_files.sh local_200_none_label_emb

if [ $# -eq 0 ]; then
    echo "用法: $0 <config_name>"
    echo "例如: $0 local_200_none_label_emb"
    exit 1
fi

config_name=$1

echo "=== 清理配置 $config_name 的模型文件 ==="

# 要删除的文件模式
files_to_delete=(
    "checkpoints/model_${config_name}.pt"
    "checkpoints/model_${config_name}_task_heads.pt"
    "checkpoints/label_embedding_${config_name}.pt"
    "checkpoints/train_info_${config_name}.json"
)

# 删除文件
deleted_count=0
for file in "${files_to_delete[@]}"; do
    if [ -f "$file" ]; then
        echo "删除: $file"
        rm -f "$file"
        deleted_count=$((deleted_count + 1))
    else
        echo "文件不存在: $file"
    fi
done

echo "清理完成，删除了 $deleted_count 个文件"

# 显示剩余空间
echo ""
echo "=== 磁盘使用情况 ==="
df -h . 