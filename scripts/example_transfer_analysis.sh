#!/bin/bash

# 多模态持续学习0样本转移分析示例

echo "=== 多模态持续学习0样本转移分析 ==="

# 1. 分析训练结果
echo "1. 分析训练结果..."
python scripts/analyze_transfer.py \
    --train_info checkpoints/train_info_twitter2015-200_none_multi.json \
    --output_dir ./analysis_results/none_multi \
    --plot

# 2. 对比不同策略的转移效果
echo "2. 对比不同策略的转移效果..."

# EWC策略
python scripts/analyze_transfer.py \
    --train_info checkpoints/train_info_twitter2015-200_ewc_multi.json \
    --output_dir ./analysis_results/ewc_multi \
    --plot

# Replay策略
python scripts/analyze_transfer.py \
    --train_info checkpoints/train_info_twitter2015-200_replay_multi.json \
    --output_dir ./analysis_results/replay_multi \
    --plot

# 3. 生成对比报告
echo "3. 生成对比报告..."
python -c "
import json
import os
from pathlib import Path

# 收集所有分析结果
results = {}
for strategy in ['none', 'ewc', 'replay']:
    result_file = f'./analysis_results/{strategy}_multi/transfer_analysis.json'
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            results[strategy] = json.load(f)

# 生成对比报告
if results:
    print('=' * 60)
    print('策略对比报告')
    print('=' * 60)
    
    metrics = ['ZS_ACC', 'FWT', 'BWT', 'AA', 'text_task_transfer', 'ner_task_transfer']
    
    for metric in metrics:
        print(f'\n{metric}:')
        print('-' * 20)
        for strategy, data in results.items():
            value = data.get('transfer_metrics', {}).get(metric, 'N/A')
            print(f'{strategy:10s}: {value}')
"

echo "分析完成！"
echo "结果保存在 ./analysis_results/ 目录下" 