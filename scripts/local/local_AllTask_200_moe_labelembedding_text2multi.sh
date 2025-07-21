source /d/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
#!/bin/bash

# LOCAL - ALL TASKS - 200 - MOE_LABELEMBEDDING - TEXT2MULTI
# MoE + Label Embedding - 先执行text_only，再执行multimodal

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 设置路径
BASE_DIR="./"
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
LABEL_EMB_PATH="$CHECKPOINT_DIR/label_embedding_200.pt"

echo "=== Starting text2multi training on 200 with MOE_LABELEMBEDDING ==="
echo "Environment: LOCAL"
echo "Tasks: masc, mate, mner, mabsa"
echo "Mode: First text_only, then multimodal"

# 训练任务列表
TASKS=('masc' 'mate' 'mner' 'mabsa')
i=0

FINAL_MODEL="$CHECKPOINT_DIR/200_moe_labelembedding_t2m.pt"
TMP_MODEL="$CHECKPOINT_DIR/1.pt"
PREV_MODEL=""

# 第一轮：text_only模式
echo "=== ROUND 1: TEXT_ONLY MODE ==="
for (( ; i<${#TASKS[@]}; i++ )); do
    TASK_NAME="${TASKS[$i]}"
    SESSION_NAME="session_$((i+1))_${TASK_NAME}_200_moe_labelembedding_text"
    
    # 根据任务确定实际数据集
    if [ "$TASK_NAME" = "mner" ]; then
        ACTUAL_DATASET="200_ner"
        DATASET_NAME="200"
        DATA_DIR="./data"
    else
        ACTUAL_DATASET="200"
        DATASET_NAME="200"
        DATA_DIR="./data"
    fi
    
    MODEL_PATH="$TMP_MODEL"
    
    PRETRAINED_OPT=""
    if [ -n "$PREV_MODEL" ]; then
        PRETRAINED_OPT="--pretrained_model_path $PREV_MODEL"
    fi
    
    TRAIN_INFO_PATH="$CHECKPOINT_DIR/train_info_200_moe_labelembedding_t2m.json"
    LOG_FILE="$LOG_DIR/$TASK_NAME_200_moe_labelembedding_text.log"
    
    # 获取数据集文件路径
    if [ "$TASK_NAME" = "mner" ]; then
        TRAIN_FILE="data/MNER/200/train.txt"
        DEV_FILE="data/MNER/200/dev.txt"
        TEST_FILE="data/MNER/200/test.txt"
        if [ "200" = "200" ]; then
            TRAIN_FILE="data/MNER/twitter2015/train__.txt"
            DEV_FILE="data/MNER/twitter2015/dev__.txt"
            TEST_FILE="data/MNER/twitter2015/test__.txt"
        fi
    else
        TRAIN_FILE="data/MASC/200/train.txt"
        DEV_FILE="data/MASC/200/dev.txt"
        TEST_FILE="data/MASC/200/test.txt"
        if [ "200" = "200" ]; then
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
    
    echo "=== Training task $((i+1))/8: $TASK_NAME (TEXT_ONLY) ==="
    echo "Task: $TASK_NAME, Dataset: 200 (actual: $ACTUAL_DATASET)"
    
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
        --moe_adapters 1 \
        --moe_num_experts 4 \
        --moe_top_k 2 \
        --use_label_embedding 1 \
        --use_label_embedding \
        --label_emb_dim 128 \
        --use_similarity_reg \
        --similarity_weight 0.1 \
        --label_embedding_path $LABEL_EMB_PATH \
        --moe_adapters \
        --moe_num_experts 4 \
        --moe_top_k 2 \
        --epochs 1 \
        --batch_size 16 \
        --lr 5e-05 \
        --weight_decay 1e-5 \
        --num_workers 4
    
    echo "=== Task $TASK_NAME (TEXT_ONLY) completed ==="
    # 保存本轮输出，供下一轮使用
    PREV_MODEL=$TMP_MODEL
done

echo "=== TEXT_ONLY ROUND COMPLETED ==="
PREV_MODEL_TEXT=$PREV_MODEL   # 记住 text 阶段最后一个模型

# 第二轮：multimodal模式
echo "=== ROUND 2: MULTIMODAL MODE ==="
i=0
PREV_MODEL=$PREV_MODEL_TEXT
for (( ; i<${#TASKS[@]}; i++ )); do
    TASK_NAME="${TASKS[$i]}"
    SESSION_NAME="session_$((i+5))_${TASK_NAME}_200_moe_labelembedding_multi"
    
    # 根据任务确定实际数据集
    if [ "$TASK_NAME" = "mner" ]; then
        ACTUAL_DATASET="200_ner"
        DATASET_NAME="200"
        DATA_DIR="./data"
    else
        ACTUAL_DATASET="200"
        DATASET_NAME="200"
        DATA_DIR="./data"
    fi
    
    if [ $i -eq $(( ${#TASKS[@]} - 1 )) ]; then
        MODEL_PATH="$FINAL_MODEL"
    else
        MODEL_PATH="$TMP_MODEL"
    fi
    
    # 设置预训练模型路径：
    if [ -n "$PREV_MODEL" ]; then
        PRETRAINED_OPT="--pretrained_model_path $PREV_MODEL"
    fi
    
    TRAIN_INFO_PATH="$CHECKPOINT_DIR/train_info_200_moe_labelembedding_t2m.json"
    LOG_FILE="$LOG_DIR/$TASK_NAME_200_moe_labelembedding_multi.log"
    
    # 获取数据集文件路径
    if [ "$TASK_NAME" = "mner" ]; then
        TRAIN_FILE="data/MNER/200/train.txt"
        DEV_FILE="data/MNER/200/dev.txt"
        TEST_FILE="data/MNER/200/test.txt"
        if [ "200" = "200" ]; then
            TRAIN_FILE="data/MNER/twitter2015/train__.txt"
            DEV_FILE="data/MNER/twitter2015/dev__.txt"
            TEST_FILE="data/MNER/twitter2015/test__.txt"
        fi
    else
        TRAIN_FILE="data/MASC/200/train.txt"
        DEV_FILE="data/MASC/200/dev.txt"
        TEST_FILE="data/MASC/200/test.txt"
        if [ "200" = "200" ]; then
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
    
    echo "=== Training task $((i+5))/8: $TASK_NAME (MULTIMODAL) ==="
    echo "Task: $TASK_NAME, Dataset: 200 (actual: $ACTUAL_DATASET)"
    
    # 执行训练
    python -m scripts.train_main \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --data_dir $DATA_DIR \
        --session_name $SESSION_NAME \
        $PRETRAINED_OPT \
        --output_model_path $MODEL_PATH \
        --train_info_json $TRAIN_INFO_PATH \
        --ewc_dir $EWC_DIR \
        --gem_mem_dir $GEM_DIR \
        --log_file $LOG_FILE \
        --train_text_file $TRAIN_FILE \
        --dev_text_file $DEV_FILE \
        --test_text_file $TEST_FILE \
        --num_labels $NUM_LABELS \
        --mode multimodal \
        --moe_adapters 1 \
        --moe_num_experts 4 \
        --moe_top_k 2 \
        --use_label_embedding 1 \
        --use_label_embedding \
        --label_emb_dim 128 \
        --use_similarity_reg \
        --similarity_weight 0.1 \
        --label_embedding_path $LABEL_EMB_PATH \
        --moe_adapters \
        --moe_num_experts 4 \
        --moe_top_k 2 \
        --epochs 1 \
        --batch_size 16 \
        --lr 5e-05 \
        --weight_decay 1e-5 \
        --num_workers 4
    
    echo "=== Task $TASK_NAME (MULTIMODAL) completed ==="
    PREV_MODEL=$MODEL_PATH
done

echo "=== All tasks completed ==="
echo "Final model saved to: $MODEL_PATH"
echo "Training info saved to: $TRAIN_INFO_PATH"
