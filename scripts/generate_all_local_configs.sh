#!/bin/bash

# 本地版本 - 生成所有策略的配置文件
# 包含所有持续学习策略，每个策略都有使用和不使用label_embedding两个版本

echo "=== 生成本地版本的所有配置文件 ==="

# 定义所有策略
strategies=("none" "ewc" "replay" "lwf" "si" "mas" "gem" "mymethod" "tam_cl" "moe" "clap4clip")

# 定义任务序列和模式序列
task_sequence=("masc" "mate" "mner" "mabsa" "masc" "mate" "mner" "mabsa")
mode_sequence=("text_only" "text_only" "text_only" "text_only" "multimodal" "multimodal" "multimodal" "multimodal")

# 环境设置为local
env="local"
dataset="200"

echo "环境: $env"
echo "数据集: $dataset"
echo "任务序列: ${task_sequence[*]}"
echo "模式序列: ${mode_sequence[*]}"
echo ""

# 生成所有策略的配置文件
for strategy in "${strategies[@]}"; do
    echo "生成策略: $strategy"
    
    # 不使用label_embedding的版本
    echo "  - 不使用label_embedding"
    python scripts/generate_task_config.py \
        --env $env \
        --dataset $dataset \
        --strategy $strategy \
        --task_sequence "${task_sequence[@]}" \
        --mode_sequence "${mode_sequence[@]}" \
        --output "scripts/configs/local_${dataset}_${strategy}.json"
    
    # 使用label_embedding的版本
    echo "  - 使用label_embedding"
    python scripts/generate_task_config.py \
        --env $env \
        --dataset $dataset \
        --strategy $strategy \
        --task_sequence "${task_sequence[@]}" \
        --mode_sequence "${mode_sequence[@]}" \
        --use_label_embedding \
        --output "scripts/configs/local_${dataset}_${strategy}_label_emb.json"
    
    echo ""
done

echo "=== 本地版本配置文件生成完成 ==="
echo "生成的文件位置: scripts/configs/"
echo "文件命名格式: local_${dataset}_[策略名](_label_emb).json"
echo ""
echo "生成的文件列表:"
ls -la scripts/configs/local_${dataset}_*.json 