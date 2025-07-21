#!/bin/bash
# 示例：0样本检测持续学习训练流程

echo "=== 0样本检测持续学习训练示例 ==="

# 1. 生成任务配置文件 - 示例1：text_only模式
echo "1. 生成任务配置文件（text_only模式）..."
python scripts/generate_task_config.py \
    --env local \
    --dataset 200 \
    --strategy ewc \
    --task_sequence masc mate mner mabsa \
    --mode_sequence text_only text_only text_only text_only \
    --use_label_embedding \
    --output task_config_ewc_text_only.json

echo "配置文件生成完成！"
echo ""

# 2. 查看配置文件内容
echo "2. 配置文件内容预览："
echo "任务列表："
python -c "
import json
with open('task_config_ewc_text_only.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
print(f'模式后缀: {config[\"mode_suffix\"]}')
print(f'模式序列: {config[\"mode_sequence\"]}')
for i, (task, mode) in enumerate(zip(config['tasks'], config['mode_sequence'])):
    print(f'  {i+1}. {task[\"task_name\"]} ({task[\"session_name\"]}) - {mode}')
    print(f'     数据集: {task[\"train_text_file\"]}')
    print(f'     标签数: {task[\"num_labels\"]}')
    print(f'     训练轮数: {task[\"epochs\"]}')
    print(f'     学习率: {task[\"lr\"]}')
    print(f'     批次大小: {task[\"batch_size\"]}')
    print()
"

echo "3. 开始训练（支持0样本检测）..."
echo "训练将按顺序执行所有任务，每个任务都会对后续任务进行0样本检测"
echo ""

# 3. 执行训练
python scripts/train_with_zero_shot.py \
    --config task_config_ewc_text_only.json

echo ""
echo "=== 训练完成 ==="
echo ""
echo "4. 查看训练结果："
echo "训练信息文件："
ls -la checkpoints/train_info_*.json 2>/dev/null || echo "训练信息文件未找到"

echo ""
echo "模型文件："
ls -la checkpoints/model_*.pt 2>/dev/null || echo "模型文件未找到"

echo ""
echo "5. 分析0样本转移指标："
echo "可以使用以下命令分析训练结果："
echo "python scripts/analyze_zero_shot_metrics.py --train_info checkpoints/train_info_200_ewc_t_label_emb.json"

echo ""
echo "=== 其他配置示例 ==="
echo ""
echo "生成不同模式的配置文件："
echo ""
echo "# 示例1: 混合模式 (text_only -> multimodal -> text_only -> multimodal)"
echo "python scripts/generate_task_config.py \\"
echo "    --env local \\"
echo "    --dataset 200 \\"
echo "    --strategy ewc \\"
echo "    --task_sequence masc mate mner mabsa \\"
echo "    --mode_sequence text_only multimodal text_only multimodal \\"
echo "    --output task_config_ewc_t2m.json"
echo ""
echo "# 示例2: 全multimodal模式"
echo "python scripts/generate_task_config.py \\"
echo "    --env local \\"
echo "    --dataset 200 \\"
echo "    --strategy ewc \\"
echo "    --task_sequence masc mate mner mabsa \\"
echo "    --mode_sequence multimodal multimodal multimodal multimodal \\"
echo "    --output task_config_ewc_multimodal.json"
echo ""
echo "# 示例3: 不同策略"
echo "python scripts/generate_task_config.py \\"
echo "    --env local \\"
echo "    --dataset 200 \\"
echo "    --strategy replay \\"
echo "    --task_sequence masc mate mner mabsa \\"
echo "    --mode_sequence text_only text_only multimodal multimodal \\"
echo "    --output task_config_replay_t2m.json"
echo ""
echo "# 示例4: 服务器环境"
echo "python scripts/generate_task_config.py \\"
echo "    --env server \\"
echo "    --dataset 200 \\"
echo "    --strategy ewc \\"
echo "    --task_sequence masc mate mner mabsa \\"
echo "    --mode_sequence text_only multimodal text_only multimodal \\"
echo "    --output task_config_server_t2m.json"
echo ""
echo "# 示例5: 不同数据集"
echo "python scripts/generate_task_config.py \\"
echo "    --env local \\"
echo "    --dataset twitter2017 \\"
echo "    --strategy ewc \\"
echo "    --task_sequence masc mate mner mabsa \\"
echo "    --mode_sequence text_only multimodal text_only multimodal \\"
echo "    --output task_config_twitter2017_t2m.json"
echo ""
echo "# 示例6: 完整数据集"
echo "python scripts/generate_task_config.py \\"
echo "    --env local \\"
echo "    --dataset twitter2015 \\"
echo "    --strategy ewc \\"
echo "    --task_sequence masc mate mner mabsa \\"
echo "    --mode_sequence multimodal multimodal multimodal multimodal \\"
echo "    --output task_config_twitter2015_multimodal.json"
echo ""
echo "=== 文件命名规则说明 ==="
echo ""
echo "根据模式序列，文件后缀会自动生成："
echo "  - 全text_only: 后缀为 't'"
echo "  - 全multimodal: 后缀为 'm'"
echo "  - 混合模式: 后缀为 't2m'"
echo ""
echo "例如："
echo "  - task_config_ewc_text_only.json -> model_200_ewc_t_label_emb.pt"
echo "  - task_config_ewc_multimodal.json -> model_200_ewc_m_label_emb.pt"
echo "  - task_config_ewc_t2m.json -> model_200_ewc_t2m_label_emb.pt" 