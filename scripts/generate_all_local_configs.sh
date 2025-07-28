#!/bin/bash

# 本地版本 - 生成所有策略的配置文件
# 包含所有持续学习策略，每个策略都有使用和不使用label_embedding两个版本
# 支持两种不同的任务序列顺序

echo "=== 生成本地版本的所有配置文件 ==="

# 定义所有策略
strategies=("none" "ewc" "replay" "lwf" "si" "mas" "gem" "tam_cl" "moe" "clap4clip")

# 定义两种任务序列和模式序列
# seq1: 原来的顺序
task_sequence_seq1=("masc" "mate" "mner" "mabsa" "masc" "mate" "mner" "mabsa")
mode_sequence_seq1=("text_only" "text_only" "text_only" "text_only" "multimodal" "multimodal" "multimodal" "multimodal")

# seq2: 新的顺序
task_sequence_seq2=("mate" "mner" "mabsa" "masc" "mate" "mner" "mabsa" "masc")
mode_sequence_seq2=("text_only" "text_only" "text_only" "text_only" "multimodal" "multimodal" "multimodal" "multimodal")

# 环境设置为local
env="local"
dataset="200"

echo "环境: $env"
echo "数据集: $dataset"
echo "seq1任务序列: ${task_sequence_seq1[*]}"
echo "seq1模式序列: ${mode_sequence_seq1[*]}"
echo "seq2任务序列: ${task_sequence_seq2[*]}"
echo "seq2模式序列: ${mode_sequence_seq2[*]}"
echo ""

# 生成所有策略的配置文件
for strategy in "${strategies[@]}"; do
    echo "生成策略: $strategy"
    
    # seq1版本 - 不使用label_embedding
    echo "  - seq1版本，不使用label_embedding"
    python scripts/generate_task_config.py \
        --env $env \
        --dataset $dataset \
        --strategy $strategy \
        --task_sequence "${task_sequence_seq1[@]}" \
        --mode_sequence "${mode_sequence_seq1[@]}" \
        --seq_suffix "seq1" \
        --output "scripts/configs/local_${dataset}_${strategy}_seq1.json"
    
    # seq1版本 - 使用label_embedding
    echo "  - seq1版本，使用label_embedding"
    python scripts/generate_task_config.py \
        --env $env \
        --dataset $dataset \
        --strategy $strategy \
        --task_sequence "${task_sequence_seq1[@]}" \
        --mode_sequence "${mode_sequence_seq1[@]}" \
        --seq_suffix "seq1" \
        --use_label_embedding \
        --output "scripts/configs/local_${dataset}_${strategy}_seq1_label_emb.json"
    
    # seq2版本 - 不使用label_embedding
    echo "  - seq2版本，不使用label_embedding"
    python scripts/generate_task_config.py \
        --env $env \
        --dataset $dataset \
        --strategy $strategy \
        --task_sequence "${task_sequence_seq2[@]}" \
        --mode_sequence "${mode_sequence_seq2[@]}" \
        --seq_suffix "seq2" \
        --output "scripts/configs/local_${dataset}_${strategy}_seq2.json"
    
    # seq2版本 - 使用label_embedding
    echo "  - seq2版本，使用label_embedding"
    python scripts/generate_task_config.py \
        --env $env \
        --dataset $dataset \
        --strategy $strategy \
        --task_sequence "${task_sequence_seq2[@]}" \
        --mode_sequence "${mode_sequence_seq2[@]}" \
        --seq_suffix "seq2" \
        --use_label_embedding \
        --output "scripts/configs/local_${dataset}_${strategy}_seq2_label_emb.json"
    
    echo ""
done

echo "=== 本地版本配置文件生成完成 ==="
echo "生成的文件位置: scripts/configs/"
echo "文件命名格式: local_${dataset}_[策略名]_seq[1|2](_label_emb).json"
echo ""
echo "生成的文件列表:"
ls -la scripts/configs/local_${dataset}_*_seq*.json

echo ""
echo "=== 生成统计 ==="
echo "策略数量: ${#strategies[@]}"
echo "序列版本数: 2 (seq1/seq2)"
echo "每个策略的版本数: 4 (seq1/seq2 × 有/无label_embedding)"
echo "总配置文件数量: $(( ${#strategies[@]} * 4 ))" 