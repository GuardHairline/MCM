#!/bin/bash

# 示例：使用标签嵌入进行多任务持续学习训练

# 设置基本参数
TASK_NAME="mabsa"
DATASET_NAME="twitter2015"
DATA_DIR="./data"
SESSION_NAME="session_1"
OUTPUT_DIR="./checkpoints/label_embedding_experiment"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 训练第一个任务 (MABSA)
echo "=== Training MABSA with label embedding ==="
python scripts/train_main.py \
    --task_name mabsa \
    --dataset_name twitter2015 \
    --data_dir $DATA_DIR \
    --session_name $SESSION_NAME \
    --epochs 10 \
    --batch_size 16 \
    --lr 2e-5 \
    --use_label_embedding \
    --label_emb_dim 128 \
    --use_similarity_reg \
    --similarity_weight 0.1 \
    --label_embedding_path $OUTPUT_DIR/label_embedding.pt \
    --output_model_path $OUTPUT_DIR/mabsa_model.pt \
    --train_info_json $OUTPUT_DIR/train_info.json \
    --log_file $OUTPUT_DIR/mabsa_training.log

# 训练第二个任务 (MASC)
echo "=== Training MASC with label embedding ==="
python scripts/train_main.py \
    --task_name masc \
    --dataset_name twitter2015 \
    --data_dir $DATA_DIR \
    --session_name session_2 \
    --epochs 10 \
    --batch_size 16 \
    --lr 2e-5 \
    --use_label_embedding \
    --label_emb_dim 128 \
    --use_similarity_reg \
    --similarity_weight 0.1 \
    --label_embedding_path $OUTPUT_DIR/label_embedding.pt \
    --output_model_path $OUTPUT_DIR/masc_model.pt \
    --train_info_json $OUTPUT_DIR/train_info.json \
    --log_file $OUTPUT_DIR/masc_training.log

# 训练第三个任务 (MATE)
echo "=== Training MATE with label embedding ==="
python scripts/train_main.py \
    --task_name mate \
    --dataset_name twitter2015 \
    --data_dir $DATA_DIR \
    --session_name session_3 \
    --epochs 10 \
    --batch_size 16 \
    --lr 2e-5 \
    --use_label_embedding \
    --label_emb_dim 128 \
    --use_similarity_reg \
    --similarity_weight 0.1 \
    --label_embedding_path $OUTPUT_DIR/label_embedding.pt \
    --output_model_path $OUTPUT_DIR/mate_model.pt \
    --train_info_json $OUTPUT_DIR/train_info.json \
    --log_file $OUTPUT_DIR/mate_training.log

# 训练第四个任务 (MNER)
echo "=== Training MNER with label embedding ==="
python scripts/train_main.py \
    --task_name mner \
    --dataset_name twitter2015 \
    --data_dir $DATA_DIR \
    --session_name session_4 \
    --epochs 10 \
    --batch_size 16 \
    --lr 2e-5 \
    --use_label_embedding \
    --label_emb_dim 128 \
    --use_similarity_reg \
    --similarity_weight 0.1 \
    --label_embedding_path $OUTPUT_DIR/label_embedding.pt \
    --output_model_path $OUTPUT_DIR/mner_model.pt \
    --train_info_json $OUTPUT_DIR/train_info.json \
    --log_file $OUTPUT_DIR/mner_training.log

echo "=== All tasks completed! ==="
echo "Check the results in: $OUTPUT_DIR" 