#!/bin/bash

# SERVER - ALL TASKS - TWITTER2017 - REPLAY - TEXT_ONLY
# Experience Replay

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 设置路径
BASE_DIR="/root/autodl-tmp"
CHECKPOINT_DIR="$BASE_DIR/checkpoints"
LOG_DIR="$BASE_DIR/log"
EWC_DIR="$BASE_DIR/ewc_params"
GEM_DIR="$BASE_DIR/gem_memory"

# 创建目录
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR
mkdir -p $EWC_DIR
mkdir -p $GEM_DIR

# 标签嵌入配置

echo "=== Starting multi-task training on TWITTER2017 with REPLAY ==="
echo "Environment: SERVER"
echo "Tasks: masc, mate, mner, mabsa"

# 训练任务列表
TASKS=('masc' 'mate' 'mner' 'mabsa')
i=0

PREV_MODEL=""
FINAL_MODEL="$CHECKPOINT_DIR/twitter2017_replay_text_only.pt"
TMP_MODEL="$CHECKPOINT_DIR/1.pt"

# 逐个训练任务
for (( ; i<${#TASKS[@]}; i++ )); do
    TASK_NAME="${TASKS[$i]}"
    SESSION_NAME="session_$((i+1))_${TASK_NAME}_twitter2017_replay"
    
    # 根据任务确定实际数据集
    if [ "$TASK_NAME" = "mner" ]; then
        ACTUAL_DATASET="twitter2017_ner"
        DATASET_NAME="twitter2017"
        DATA_DIR="./data"
    else
        ACTUAL_DATASET="twitter2017"
        DATASET_NAME="twitter2017"
        DATA_DIR="./data"
    fi
    
    if [ $i -eq $(( ${#TASKS[@]} - 1 )) ]; then
        MODEL_PATH="$FINAL_MODEL"
    else
        MODEL_PATH="$TMP_MODEL"
    fi
    
    # ---------- 连续学习：始终衔接上一轮输出 ----------
    PRETRAINED_OPT=""
    if [ -n "$PREV_MODEL" ]; then
        PRETRAINED_OPT="--pretrained_model_path $PREV_MODEL"
    fi
    
    TRAIN_INFO_PATH="$CHECKPOINT_DIR/train_info_twitter2017_replay_text_only.json"
    LOG_FILE="$LOG_DIR/$TASK_NAME_twitter2017_replay_text_only.log"
    
    # 获取数据集文件路径
    if [ "$TASK_NAME" = "mner" ]; then
        TRAIN_FILE="data/MNER/twitter2017/train.txt"
        DEV_FILE="data/MNER/twitter2017/dev.txt"
        TEST_FILE="data/MNER/twitter2017/test.txt"
        if [ "twitter2017" = "200" ]; then
            TRAIN_FILE="data/MNER/twitter2015/train__.txt"
            DEV_FILE="data/MNER/twitter2015/dev__.txt"
            TEST_FILE="data/MNER/twitter2015/test__.txt"
        fi
    else
        TRAIN_FILE="data/MASC/twitter2017/train.txt"
        DEV_FILE="data/MASC/twitter2017/dev.txt"
        TEST_FILE="data/MASC/twitter2017/test.txt"
        if [ "twitter2017" = "200" ]; then
            TRAIN_FILE="data/MASC/twitter2015/train__.txt"
            DEV_FILE="data/MASC/twitter2015/dev__.txt"
            TEST_FILE="data/MASC/twitter2015/test__.txt"
        fi
    fi
    
    # 获取类别数
    case $TASK_NAME in
        mabsa)
            NUM_LABELS=7
            ;;
        masc)
            NUM_LABELS=3
            ;;
        mate)
            NUM_LABELS=3
            ;;
        mner)
            NUM_LABELS=9
            ;;
        *)
            NUM_LABELS=3
            ;;
    esac
    
    echo "=== Training task $((i+1))/4: $TASK_NAME ==="
    echo "Task: $TASK_NAME, Dataset: twitter2017 (actual: $ACTUAL_DATASET)"
    
    # 执行训练
    python -m scripts.train_main \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --data_dir $DATA_DIR \
        --session_name $SESSION_NAME \
        --output_model_path $MODEL_PATH \
        --train_info_json $TRAIN_INFO_PATH \
        --ewc_dir $EWC_DIR \
        --gem_mem_dir $GEM_DIR \
        --log_file $LOG_FILE \
        --train_text_file $TRAIN_FILE \
        --dev_text_file $DEV_FILE \
        --test_text_file $TEST_FILE \
        --num_labels $NUM_LABELS \
        --mode text_only \
        $PRETRAINED_OPT \
        --replay 1 \
        --memory_percentage 0.05 \
        --replay_ratio 0.5 \
        --replay_frequency 4 \
        --epochs 20 \
        --batch_size 16 \
        --lr 5e-05 \
        --weight_decay 1e-05 \
        --step_size 10 \
        --gamma 0.5 \
        --dropout_prob 0.1 \
        --num_workers 4
    
    echo "=== Task $TASK_NAME completed ==="
    # 保存本轮输出，供下一轮使用
    PREV_MODEL=$TMP_MODEL
done

echo "=== All tasks completed ==="
echo "Final model saved to: $FINAL_MODEL"
echo "Training info saved to: $TRAIN_INFO_PATH"

shutdown -h now
