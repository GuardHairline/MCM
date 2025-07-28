#!/usr/bin/env python3
# scripts/config_templates.py
"""
配置模板生成器
用于生成不同环境的训练脚本
"""

import os
import argparse
from typing import Dict, Any, List


class ConfigTemplate:
    """配置模板类"""
    
    def __init__(self):
        # 环境配置
        self.environments = {
            "server": {
                "base_dir": "/root/autodl-tmp",
                "model_name": "1.pt",  # 服务器版本统一命名
                "log_dir": "/root/autodl-tmp/log",
                "checkpoint_dir": "/root/autodl-tmp/checkpoints",
                "ewc_dir": "/root/autodl-tmp/ewc_params",
                "gem_dir": "/root/autodl-tmp/gem_memory"
            },
            "local": {
                "base_dir": "./",
                "model_name": "{task}_{dataset}_{strategy}.pt",  # 本地版本详细命名
                "log_dir": "./log",
                "checkpoint_dir": "./checkpoints",
                "ewc_dir": "./ewc_params",
                "gem_dir": "./gem_memory"
            }
        }
        
        # 数据集配置
        self.datasets = {
            "twitter2015": {
                "data_dir": "./data",
                "dataset_name": "twitter2015",
                "full_files": {
                    "train": "data/MASC/twitter2015/train.txt",
                    "dev": "data/MASC/twitter2015/dev.txt",
                    "test": "data/MASC/twitter2015/test.txt"
                },
                "simplified_files": {
                    "train": "data/MASC/twitter2015/train__.txt",
                    "dev": "data/MASC/twitter2015/dev__.txt",
                    "test": "data/MASC/twitter2015/test__.txt"
                }
            },
            "twitter2017": {
                "data_dir": "./data",
                "dataset_name": "twitter2017",
                "full_files": {
                    "train": "data/MASC/twitter2017/train.txt",
                    "dev": "data/MASC/twitter2017/dev.txt",
                    "test": "data/MASC/twitter2017/test.txt"
                },
                "simplified_files": {
                    "train": "data/MASC/twitter2017/train__.txt",
                    "dev": "data/MASC/twitter2017/dev__.txt",
                    "test": "data/MASC/twitter2017/test__.txt"
                }
            },
            "200": {
                "data_dir": "./data",
                "dataset_name": "twitter2015",
                "files": {
                    "train": "data/MASC/twitter2015/train__.txt",
                    "dev": "data/MASC/twitter2015/dev__.txt",
                    "test": "data/MASC/twitter2015/test__.txt"
                }
            }
        }
        
        # 任务-数据集映射配置
        self.task_dataset_mapping = {
            "mabsa": {
                "twitter2015": "twitter2015",
                "twitter2017": "twitter2017",
                "200": "200"
            },
            "masc": {
                "twitter2015": "twitter2015", 
                "twitter2017": "twitter2017",
                "200": "200"
            },
            "mate": {
                "twitter2015": "twitter2015",
                "twitter2017": "twitter2017", 
                "200": "200"
            },
            "mner": {
                "twitter2015": "twitter2015_ner",  # MNER使用单独的NER数据集
                "twitter2017": "twitter2017_ner",  # MNER使用单独的NER数据集
                "200": "200_ner"  # MNER使用单独的NER数据集
            }
        }
        
        # MNER专用数据集配置
        self.mner_datasets = {
            "twitter2015_ner": {
                "data_dir": "./data",
                "dataset_name": "twitter2015",
                "full_files": {
                    "train": "data/MNER/twitter2015/train.txt",
                    "dev": "data/MNER/twitter2015/dev.txt", 
                    "test": "data/MNER/twitter2015/test.txt"
                },
                "simplified_files": {
                    "train": "data/MNER/twitter2015/train__.txt",
                    "dev": "data/MNER/twitter2015/dev__.txt",
                    "test": "data/MNER/twitter2015/test__.txt"
                }
            },
            "twitter2017_ner": {
                "data_dir": "./data",
                "dataset_name": "twitter2017",
                "full_files": {
                    "train": "data/MNER/twitter2017/train.txt",
                    "dev": "data/MNER/twitter2017/dev.txt",
                    "test": "data/MNER/twitter2017/test.txt"
                },
                "simplified_files": {
                    "train": "data/MNER/twitter2017/train__.txt",
                    "dev": "data/MNER/twitter2017/dev__.txt",
                    "test": "data/MNER/twitter2017/test__.txt"
                }
            },
            "200_ner": {
                "data_dir": "./data",
                "dataset_name": "twitter2015",
                "files": {
                    "train": "data/MNER/twitter2015/train__.txt",
                    "dev": "data/MNER/twitter2015/dev__.txt",
                    "test": "data/MNER/twitter2015/test__.txt"
                }
            }
        }
        
        # 持续学习策略配置
        self.strategies = {
            "none": {
                "params": {},
                "description": "无持续学习策略"
            },
            "ewc": {
                "params": {
                    "ewc": 1,
                    "ewc_lambda": 1000.0
                },
                "description": "Elastic Weight Consolidation"
            },
            "replay": {
                "params": {
                    "replay": 1,
                    "memory_percentage": 0.05,
                    "replay_ratio": 0.5,
                    "replay_frequency": 4
                },
                "description": "Experience Replay"
            },
            "lwf": {
                "params": {
                    "lwf": 1,
                    "lwf_T": 2.0,
                    "lwf_alpha": 0.5,
                    "lwf_decay": 0.5
                },
                "description": "Learning without Forgetting"
            },
            "si": {
                "params": {
                    "si": 1,
                    "si_epsilon": 0.1,
                    "si_decay": 0.5
                },
                "description": "Synaptic Intelligence"
            },
            "mas": {
                "params": {
                    "mas": 1,
                    "mas_eps": 1e-3,
                    "mas_decay": 0.5
                },
                "description": "Memory Aware Synapses"
            },
            "gem": {
                "params": {
                    "gem": 1,
                    "gem_mem": 100
                },
                "description": "Gradient Episodic Memory"
            },
            "mymethod": {
                "params": {
                    "mymethod": 1,
                    "ewc_lambda": 1000.0
                },
                "description": "自定义方法"
            },
            "tamcl": {
                "params": {
                    "tam_cl": 1
                },
                "description": "TAM-CL"
            },
            "moe": {
                "params": {
                    "moe_adapters": 1,
                    "moe_num_experts": 4,
                    "moe_top_k": 2
                },
                "description": "MoE Adapters"
            },
            "clap4clip": {
                "params": {
                    "clap4clip": 1
                },
                "description": "CLAP4CLIP"
            },
            "labelembedding": {
                "params": {
                    "use_label_embedding": 1,
                    "use_hierarchical_head": 1
                },
                "description": "Label Embedding with Hierarchical Head"
            },
            "moe_labelembedding": {
                "params": {
                    "moe_adapters": 1,
                    "moe_num_experts": 4,
                    "moe_top_k": 2,
                    "use_label_embedding": 1,
                    "use_hierarchical_head": 1
                },
                "description": "MoE + Label Embedding"
            },
            "clap4clip_labelembedding": {
                "params": {
                    "clap4clip": 1,
                    "use_label_embedding": 1,
                    "use_hierarchical_head": 1
                },
                "description": "CLAP4CLIP + Label Embedding"
            }
        }
        
        # 任务配置
        self.tasks = {
            "AllTask": {
                "tasks": ["masc", "mate", "mner", "mabsa"],
                "description": "多任务训练"
            },
            "SingleTask": {
                "tasks": ["mabsa"],  # 可配置
                "description": "单任务训练"
            }
        }
    
    def generate_script_name(self, env: str, task_type: str, dataset: str, 
                           strategy: str, mode: str = "multi", use_label_embedding: bool = False, task: str = None) -> str:
        """
        生成脚本文件名，若 use_label_embedding=True，文件名加 _labelembedding
        """
        base_name = f"{env}_{task_type}_{dataset}_{strategy}_{mode}"
        # 只有策略名本身不含 labelembedding 时才加后缀
        if use_label_embedding and "labelembedding" not in strategy:
            base_name += "_labelembedding"
        return base_name + ".sh"
    
    def generate_single_task_script(self, env: str, task: str, dataset: str, 
                                  strategy: str, mode: str = "multi", 
                                  use_label_embedding: bool = False, use_clap4clip: bool = False, use_moe_adapters: bool = False, epochs: int = None, lr: float = None) -> str:
        """生成单任务训练脚本"""
        env_config = self.environments[env]
        strategy_config = self.strategies[strategy]
        
        # 根据任务和数据集确定实际使用的数据集配置
        actual_dataset = self.task_dataset_mapping[task][dataset]
        if task == "mner":
            dataset_config = self.mner_datasets[actual_dataset]
        else:
            dataset_config = self.datasets[dataset]
        
        # 确定模型文件名
        if env == "server":
            model_name = "1.pt"
        else:
            model_name = f"{task}_{dataset}_{strategy}.pt"
        
        # 获取数据集文件路径
        files = self.get_text_files(task, dataset)
        # 获取类别数
        task_num_labels = {
            "mabsa": 7,
            "masc": 3,
            "mate": 3,
            "mner": 9
        }
        num_labels = task_num_labels.get(task, 3)
        
        # 构建脚本内容
        script_content = f"""#!/bin/bash

# {env.upper()} - {task.upper()} - {dataset.upper()} - {strategy.upper()} - {mode.upper()}
# {strategy_config['description']}

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 设置路径
BASE_DIR="{env_config['base_dir']}"
CHECKPOINT_DIR="$BASE_DIR/checkpoints"
LOG_DIR="$BASE_DIR/log"
EWC_DIR="$BASE_DIR/ewc_params"
GEM_DIR="$BASE_DIR/gem_memory"

# 创建目录
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR
mkdir -p $EWC_DIR
mkdir -p $GEM_DIR

# 训练参数
TASK_NAME="{task}"
DATASET_NAME="{dataset_config['dataset_name']}"
DATA_DIR="{dataset_config['data_dir']}"
SESSION_NAME="session_{task}_{dataset}_{strategy}"
MODEL_PATH="$CHECKPOINT_DIR/{model_name}"
TRAIN_INFO_PATH="$CHECKPOINT_DIR/train_info_{task}_{dataset}_{strategy}_{mode}.json"
LOG_FILE="$LOG_DIR/{task}_{dataset}_{strategy}_{mode}.log"

echo "=== Starting {task.upper()} training on {dataset.upper()} with {strategy.upper()} ==="
echo "Environment: {env.upper()}"
echo "Task: {task.upper()}"
echo "Dataset: {dataset.upper()} (actual: {actual_dataset})"
echo "Model will be saved to: $MODEL_PATH"
echo "Log will be saved to: $LOG_FILE"

# 执行训练
python -m scripts.train_main \\
    --task_name $TASK_NAME \\
    --dataset_name $DATASET_NAME \\
    --data_dir $DATA_DIR \\
    --session_name $SESSION_NAME \\
    --output_model_path $MODEL_PATH \\
    --train_info_json $TRAIN_INFO_PATH \\
    --ewc_dir $EWC_DIR \\
    --gem_mem_dir $GEM_DIR \\
    --log_file $LOG_FILE \\
    --train_text_file {files['train']} \\
    --dev_text_file {files['dev']} \\
    --test_text_file {files['test']} \\
    --num_labels {num_labels} \\
    --mode {mode} \\
"""
        
        # 添加策略参数
        for param, value in strategy_config['params'].items():
            script_content += f"    --{param} {value} \\\n"
        
        # 添加标签嵌入参数
        if use_label_embedding:
            script_content += f"""    --use_label_embedding \\
    --label_emb_dim 128 \\
    --use_similarity_reg \\
    --similarity_weight 0.1 \\
    --label_embedding_path $CHECKPOINT_DIR/label_embedding_{task}_{dataset}.pt \\
"""
        
        # 添加 CLAP4CLIP 参数
        if use_clap4clip:
            script_content += f"    --clap4clip \\\n"
        # 添加 MoEAdapters 参数
        if use_moe_adapters:
            script_content += f"    --moe_adapters \\\n"
            script_content += f"    --moe_num_experts 4 \\\n"
            script_content += f"    --moe_top_k 2 \\\n"
        
        # 添加其他参数
        # 本地环境默认使用1个epoch用于测试，服务器环境使用20个epoch
        default_epochs = 1 if env == "local" else 20
        default_batch_size = 2 if env == "local" else 16
        default_step_size = 10
        default_gamma = 0.5
        default_weight_decay = 1e-5
        default_dropout_prob = 0.1
        script_content += f"""    --epochs {epochs if epochs is not None else default_epochs} \\
    --batch_size {default_batch_size} \\
    --lr {lr if lr is not None else 5e-5} \\
    --weight_decay {default_weight_decay} \\
    --step_size {default_step_size} \\
    --gamma {default_gamma} \\
    --dropout_prob {default_dropout_prob} \\
    --num_workers 4

echo "=== Training completed ==="
echo "Model saved to: $MODEL_PATH"
echo "Training info saved to: $TRAIN_INFO_PATH"
"""
        
        if env == "local":
            script_content = (
                "source /d/ProgramData/anaconda3/etc/profile.d/conda.sh\n"
                "conda activate pytorch\n"
            ) + script_content
        
        # 末尾追加关机命令（仅server环境）
        if env == "server":
            script_content += "\nshutdown -h now\n"
        return script_content
    
    def generate_multi_task_script(self, env: str, dataset: str, strategy: str, 
                                 mode: str = "multi", use_label_embedding: bool = False, use_clap4clip: bool = False, use_moe_adapters: bool = False, epochs: int = None, lr: float = None) -> str:
        """生成多任务训练脚本"""
        env_config = self.environments[env]
        strategy_config = self.strategies[strategy]
        tasks = self.tasks["AllTask"]["tasks"]
        
        # 处理text2multi模式：需要生成两轮训练
        if mode == "text2multi":
            return self._generate_text2multi_script(env, dataset, strategy, use_label_embedding, use_clap4clip, use_moe_adapters, epochs, lr)
        
        script_content = f"""#!/bin/bash

# {env.upper()} - ALL TASKS - {dataset.upper()} - {strategy.upper()} - {mode.upper()}
# {strategy_config['description']}

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 设置路径
BASE_DIR=\"{env_config['base_dir']}\"
CHECKPOINT_DIR=\"$BASE_DIR/checkpoints\"
LOG_DIR=\"$BASE_DIR/log\"
EWC_DIR=\"$BASE_DIR/ewc_params\"
GEM_DIR=\"$BASE_DIR/gem_memory\"

# 创建目录
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR
mkdir -p $EWC_DIR
mkdir -p $GEM_DIR

# 标签嵌入配置
"""
        if use_label_embedding:
            script_content += f"""LABEL_EMB_PATH="$CHECKPOINT_DIR/label_embedding_{dataset}.pt"
"""
        
        script_content += f"""
echo "=== Starting multi-task training on {dataset.upper()} with {strategy.upper()} ==="
echo "Environment: {env.upper()}"
echo "Tasks: {', '.join(tasks)}"

# 训练任务列表
TASKS=({ ' '.join([f"'{t}'" for t in tasks]) })
i=0

PREV_MODEL=""
FINAL_MODEL="$CHECKPOINT_DIR/{dataset}_{strategy}_{mode}.pt"
TMP_MODEL="$CHECKPOINT_DIR/1.pt"

# 逐个训练任务
for (( ; i<${{#TASKS[@]}}; i++ )); do
    TASK_NAME="${{TASKS[$i]}}"
    SESSION_NAME="session_$((i+1))_${{TASK_NAME}}_{dataset}_{strategy}"
    
    # 根据任务确定实际数据集
    if [ "$TASK_NAME" = "mner" ]; then
        ACTUAL_DATASET="{dataset}_ner"
        DATASET_NAME="{dataset}"
        DATA_DIR="./data"
    else
        ACTUAL_DATASET="{dataset}"
        DATASET_NAME="{dataset}"
        DATA_DIR="./data"
    fi
    
    if [ $i -eq $(( ${{#TASKS[@]}} - 1 )) ]; then
        MODEL_PATH="$FINAL_MODEL"
    else
        MODEL_PATH="$TMP_MODEL"
    fi
    
    # ---------- 连续学习：始终衔接上一轮输出 ----------
    PRETRAINED_OPT=""
    if [ -n "$PREV_MODEL" ]; then
        PRETRAINED_OPT="--pretrained_model_path $PREV_MODEL"
    fi
    
    TRAIN_INFO_PATH="$CHECKPOINT_DIR/train_info_{dataset}_{strategy}_{mode}.json"
    LOG_FILE="$LOG_DIR/$TASK_NAME_{dataset}_{strategy}_{mode}.log"
    
    # 获取数据集文件路径
    if [ "$TASK_NAME" = "mner" ]; then
        TRAIN_FILE="data/MNER/{dataset}/train.txt"
        DEV_FILE="data/MNER/{dataset}/dev.txt"
        TEST_FILE="data/MNER/{dataset}/test.txt"
        if [ "{dataset}" = "200" ]; then
            TRAIN_FILE="data/MNER/twitter2015/train__.txt"
            DEV_FILE="data/MNER/twitter2015/dev__.txt"
            TEST_FILE="data/MNER/twitter2015/test__.txt"
        fi
    else
        TRAIN_FILE="data/MASC/{dataset}/train.txt"
        DEV_FILE="data/MASC/{dataset}/dev.txt"
        TEST_FILE="data/MASC/{dataset}/test.txt"
        if [ "{dataset}" = "200" ]; then
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
    echo "Task: $TASK_NAME, Dataset: {dataset} (actual: $ACTUAL_DATASET)"
    
    # 执行训练
    python -m scripts.train_main \\
        --task_name $TASK_NAME \\
        --dataset_name $DATASET_NAME \\
        --data_dir $DATA_DIR \\
        --session_name $SESSION_NAME \\
        --output_model_path $MODEL_PATH \\
        --train_info_json $TRAIN_INFO_PATH \\
        --ewc_dir $EWC_DIR \\
        --gem_mem_dir $GEM_DIR \\
        --log_file $LOG_FILE \\
        --train_text_file $TRAIN_FILE \\
        --dev_text_file $DEV_FILE \\
        --test_text_file $TEST_FILE \\
        --num_labels $NUM_LABELS \\
        --mode {mode} \\
        $PRETRAINED_OPT \\
"""
        # 添加策略参数
        for param, value in strategy_config['params'].items():
            script_content += f"        --{param} {value} \\\n"
        
        # 添加标签嵌入参数
        if use_label_embedding:
            script_content += f"""        --use_label_embedding \\
        --label_emb_dim 128 \\
        --use_similarity_reg \\
        --similarity_weight 0.1 \\
        --label_embedding_path $LABEL_EMB_PATH \\
"""
        
        # 添加 CLAP4CLIP 参数
        if use_clap4clip:
            script_content += f"        --clap4clip \\\n"
        # 添加 MoEAdapters 参数
        if use_moe_adapters:
            script_content += f"        --moe_adapters \\\n"
            script_content += f"        --moe_num_experts 4 \\\n"
            script_content += f"        --moe_top_k 2 \\\n"
        
        # 添加其他参数
        # 本地环境默认使用1个epoch用于测试，服务器环境使用20个epoch
        default_epochs = 1 if env == "local" else 20
        default_batch_size = 2 if env == "local" else 16
        default_step_size = 10
        default_gamma = 0.5
        default_weight_decay = 1e-5
        default_dropout_prob = 0.1
        script_content += f"""        --epochs {epochs if epochs is not None else default_epochs} \\
        --batch_size {default_batch_size} \\
        --lr {lr if lr is not None else 5e-5} \\
        --weight_decay {default_weight_decay} \\
        --step_size {default_step_size} \\
        --gamma {default_gamma} \\
        --dropout_prob {default_dropout_prob} \\
        --num_workers 4
    
    echo "=== Task $TASK_NAME completed ==="
    # 保存本轮输出，供下一轮使用
    PREV_MODEL=$TMP_MODEL
done

echo "=== All tasks completed ==="
echo "Final model saved to: $FINAL_MODEL"
echo "Training info saved to: $TRAIN_INFO_PATH"
"""
        
        if env == "local":
            script_content = (
                "source /d/ProgramData/anaconda3/etc/profile.d/conda.sh\n"
                "conda activate pytorch\n"
            ) + script_content
        
        # 末尾追加关机命令（仅server环境）
        if env == "server":
            script_content += "\nshutdown -h now\n"
        return script_content
    
    def _generate_text2multi_script(self, env: str, dataset: str, strategy: str, 
                                  use_label_embedding: bool = False, use_clap4clip: bool = False, use_moe_adapters: bool = False, epochs: int = None, lr: float = None) -> str:
        """生成text2multi模式的脚本：先执行text_only，再执行multimodal"""
        env_config = self.environments[env]
        strategy_config = self.strategies[strategy]
        tasks = self.tasks["AllTask"]["tasks"]
        mode = "t2m"
        script_content = f"""#!/bin/bash

# {env.upper()} - ALL TASKS - {dataset.upper()} - {strategy.upper()} - TEXT2MULTI
# {strategy_config['description']} - 先执行text_only，再执行multimodal

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 设置路径
BASE_DIR=\"{env_config['base_dir']}\"
CHECKPOINT_DIR=\"$BASE_DIR/checkpoints\"
LOG_DIR=\"$BASE_DIR/log\"
EWC_DIR=\"$BASE_DIR/ewc_params\"
GEM_DIR=\"$BASE_DIR/gem_memory\"

# 创建目录
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR
mkdir -p $EWC_DIR
mkdir -p $GEM_DIR

# 标签嵌入配置
"""
        if use_label_embedding:
            script_content += f"""LABEL_EMB_PATH="$CHECKPOINT_DIR/label_embedding_{dataset}.pt"
"""
        
        script_content += f"""
echo "=== Starting text2multi training on {dataset.upper()} with {strategy.upper()} ==="
echo "Environment: {env.upper()}"
echo "Tasks: {', '.join(tasks)}"
echo "Mode: First text_only, then multimodal"

# 训练任务列表
TASKS=({ ' '.join([f"'{t}'" for t in tasks]) })
i=0

FINAL_MODEL="$CHECKPOINT_DIR/{dataset}_{strategy}_t2m.pt"
TMP_MODEL="$CHECKPOINT_DIR/1.pt"
PREV_MODEL=""

# 第一轮：text_only模式
echo "=== ROUND 1: TEXT_ONLY MODE ==="
for (( ; i<${{#TASKS[@]}}; i++ )); do
    TASK_NAME="${{TASKS[$i]}}"
    SESSION_NAME="session_$((i+1))_${{TASK_NAME}}_{dataset}_{strategy}_text"
    
    # 根据任务确定实际数据集
    if [ "$TASK_NAME" = "mner" ]; then
        ACTUAL_DATASET="{dataset}_ner"
        DATASET_NAME="{dataset}"
        DATA_DIR="./data"
    else
        ACTUAL_DATASET="{dataset}"
        DATASET_NAME="{dataset}"
        DATA_DIR="./data"
    fi
    
    MODEL_PATH="$TMP_MODEL"
    
    PRETRAINED_OPT=""
    if [ -n "$PREV_MODEL" ]; then
        PRETRAINED_OPT="--pretrained_model_path $PREV_MODEL"
    fi
    
    TRAIN_INFO_PATH="$CHECKPOINT_DIR/train_info_{dataset}_{strategy}_{mode}.json"
    LOG_FILE="$LOG_DIR/$TASK_NAME_{dataset}_{strategy}_text.log"
    
    # 获取数据集文件路径
    if [ "$TASK_NAME" = "mner" ]; then
        TRAIN_FILE="data/MNER/{dataset}/train.txt"
        DEV_FILE="data/MNER/{dataset}/dev.txt"
        TEST_FILE="data/MNER/{dataset}/test.txt"
        if [ "{dataset}" = "200" ]; then
            TRAIN_FILE="data/MNER/twitter2015/train__.txt"
            DEV_FILE="data/MNER/twitter2015/dev__.txt"
            TEST_FILE="data/MNER/twitter2015/test__.txt"
        fi
    else
        TRAIN_FILE="data/MASC/{dataset}/train.txt"
        DEV_FILE="data/MASC/{dataset}/dev.txt"
        TEST_FILE="data/MASC/{dataset}/test.txt"
        if [ "{dataset}" = "200" ]; then
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
    echo "Task: $TASK_NAME, Dataset: {dataset} (actual: $ACTUAL_DATASET)"
    
    # 执行训练
    python -m scripts.train_main \\
        --task_name $TASK_NAME \\
        --dataset_name $DATASET_NAME \\
        --data_dir $DATA_DIR \\
        --session_name $SESSION_NAME \\
        --output_model_path $MODEL_PATH \\
        --train_info_json $TRAIN_INFO_PATH \\
        --ewc_dir $EWC_DIR \\
        --gem_mem_dir $GEM_DIR \\
        --log_file $LOG_FILE \\
        --train_text_file $TRAIN_FILE \\
        --dev_text_file $DEV_FILE \\
        --test_text_file $TEST_FILE \\
        --num_labels $NUM_LABELS \\
        --mode text_only \\
        $PRETRAINED_OPT \\
"""
        # 添加策略参数
        for param, value in strategy_config['params'].items():
            script_content += f"        --{param} {value} \\\n"
        
        # 添加标签嵌入参数
        if use_label_embedding:
            script_content += f"""        --use_label_embedding \\
        --label_emb_dim 128 \\
        --use_similarity_reg \\
        --similarity_weight 0.1 \\
        --label_embedding_path $LABEL_EMB_PATH \\
"""
        
        # 添加 CLAP4CLIP 参数
        if use_clap4clip:
            script_content += f"        --clap4clip \\\n"
        # 添加 MoEAdapters 参数
        if use_moe_adapters:
            script_content += f"        --moe_adapters \\\n"
            script_content += f"        --moe_num_experts 4 \\\n"
            script_content += f"        --moe_top_k 2 \\\n"
        
        # 添加其他参数
        default_epochs = 1 if env == "local" else 20
        script_content += f"""        --epochs {epochs if epochs is not None else default_epochs} \\
        --batch_size 16 \\
        --lr {lr if lr is not None else 5e-5} \\
        --weight_decay 1e-5 \\
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
for (( ; i<${{#TASKS[@]}}; i++ )); do
    TASK_NAME="${{TASKS[$i]}}"
    SESSION_NAME="session_$((i+5))_${{TASK_NAME}}_{dataset}_{strategy}_multi"
    
    # 根据任务确定实际数据集
    if [ "$TASK_NAME" = "mner" ]; then
        ACTUAL_DATASET="{dataset}_ner"
        DATASET_NAME="{dataset}"
        DATA_DIR="./data"
    else
        ACTUAL_DATASET="{dataset}"
        DATASET_NAME="{dataset}"
        DATA_DIR="./data"
    fi
    
    if [ $i -eq $(( ${{#TASKS[@]}} - 1 )) ]; then
        MODEL_PATH="$FINAL_MODEL"
    else
        MODEL_PATH="$TMP_MODEL"
    fi
    
    # 设置预训练模型路径：
    if [ -n "$PREV_MODEL" ]; then
        PRETRAINED_OPT="--pretrained_model_path $PREV_MODEL"
    fi
    
    TRAIN_INFO_PATH="$CHECKPOINT_DIR/train_info_{dataset}_{strategy}_{mode}.json"
    LOG_FILE="$LOG_DIR/$TASK_NAME_{dataset}_{strategy}_multi.log"
    
    # 获取数据集文件路径
    if [ "$TASK_NAME" = "mner" ]; then
        TRAIN_FILE="data/MNER/{dataset}/train.txt"
        DEV_FILE="data/MNER/{dataset}/dev.txt"
        TEST_FILE="data/MNER/{dataset}/test.txt"
        if [ "{dataset}" = "200" ]; then
            TRAIN_FILE="data/MNER/twitter2015/train__.txt"
            DEV_FILE="data/MNER/twitter2015/dev__.txt"
            TEST_FILE="data/MNER/twitter2015/test__.txt"
        fi
    else
        TRAIN_FILE="data/MASC/{dataset}/train.txt"
        DEV_FILE="data/MASC/{dataset}/dev.txt"
        TEST_FILE="data/MASC/{dataset}/test.txt"
        if [ "{dataset}" = "200" ]; then
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
    echo "Task: $TASK_NAME, Dataset: {dataset} (actual: $ACTUAL_DATASET)"
    
    # 执行训练
    python -m scripts.train_main \\
        --task_name $TASK_NAME \\
        --dataset_name $DATASET_NAME \\
        --data_dir $DATA_DIR \\
        --session_name $SESSION_NAME \\
        $PRETRAINED_OPT \\
        --output_model_path $MODEL_PATH \\
        --train_info_json $TRAIN_INFO_PATH \\
        --ewc_dir $EWC_DIR \\
        --gem_mem_dir $GEM_DIR \\
        --log_file $LOG_FILE \\
        --train_text_file $TRAIN_FILE \\
        --dev_text_file $DEV_FILE \\
        --test_text_file $TEST_FILE \\
        --num_labels $NUM_LABELS \\
        --mode multimodal \\
"""
        # 添加策略参数
        for param, value in strategy_config['params'].items():
            script_content += f"        --{param} {value} \\\n"
        
        # 添加标签嵌入参数
        if use_label_embedding:
            script_content += f"""        --use_label_embedding \\
        --label_emb_dim 128 \\
        --use_similarity_reg \\
        --similarity_weight 0.1 \\
        --label_embedding_path $LABEL_EMB_PATH \\
"""
        
        # 添加 CLAP4CLIP 参数
        if use_clap4clip:
            script_content += f"        --clap4clip \\\n"
        # 添加 MoEAdapters 参数
        if use_moe_adapters:
            script_content += f"        --moe_adapters \\\n"
            script_content += f"        --moe_num_experts 4 \\\n"
            script_content += f"        --moe_top_k 2 \\\n"
        
        # 添加其他参数
        default_epochs = 1 if env == "local" else 20
        script_content += f"""        --epochs {epochs if epochs is not None else default_epochs} \\
        --batch_size 16 \\
        --lr {lr if lr is not None else 5e-5} \\
        --weight_decay 1e-5 \\
        --num_workers 4
    
    echo "=== Task $TASK_NAME (MULTIMODAL) completed ==="
    PREV_MODEL=$MODEL_PATH
done

echo "=== All tasks completed ==="
echo "Final model saved to: $MODEL_PATH"
echo "Training info saved to: $TRAIN_INFO_PATH"
"""
        
        if env == "local":
            script_content = (
                "source /d/ProgramData/anaconda3/etc/profile.d/conda.sh\n"
                "conda activate pytorch\n"
            ) + script_content
        
        # 末尾追加关机命令（仅server环境）
        if env == "server":
            script_content += "\nshutdown -h now\n"
        return script_content
    
    def generate_script(self, env: str, task_type: str, dataset: str, strategy: str, 
                       mode: str = "multi", use_label_embedding: bool = False, use_moe_adapters: bool = False, use_clap4clip: bool = False, task: str = None, epochs: int = None, lr: float = None) -> str:
        """生成训练脚本"""
        if task_type == "AllTask":
            return self.generate_multi_task_script(env, dataset, strategy, mode, use_label_embedding, use_clap4clip, use_moe_adapters, epochs=epochs, lr=lr)
        else:
            # 单任务，需要指定具体任务
            if task is None:
                task = "mabsa"  # 默认任务
            return self.generate_single_task_script(env, task, dataset, strategy, mode, use_label_embedding, use_clap4clip, use_moe_adapters, epochs=epochs, lr=lr)

    def get_text_files(self, task, dataset):
        """
        根据任务和数据集类型返回正确的 train/dev/test 文件路径
        """
        if task == "mner":
            if dataset == "200":
                files = {
                    "train": "data/MNER/twitter2015/train__.txt",
                    "dev": "data/MNER/twitter2015/dev__.txt",
                    "test": "data/MNER/twitter2015/test__.txt"
                }
            else:
                files = {
                    "train": f"data/MNER/{dataset}/train.txt",
                    "dev": f"data/MNER/{dataset}/dev.txt",
                    "test": f"data/MNER/{dataset}/test.txt"
                }
        else:
            if dataset == "200":
                files = {
                    "train": "data/MASC/twitter2015/train__.txt",
                    "dev": "data/MASC/twitter2015/dev__.txt",
                    "test": "data/MASC/twitter2015/test__.txt"
                }
            else:
                files = {
                    "train": f"data/MASC/{dataset}/train.txt",
                    "dev": f"data/MASC/{dataset}/dev.txt",
                    "test": f"data/MASC/{dataset}/test.txt"
                }
        return files


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成训练脚本")
    parser.add_argument("--env", type=str, required=True, choices=["server", "local"],
                       help="环境类型")
    parser.add_argument("--task_type", type=str, required=True, choices=["AllTask", "SingleTask"],
                       help="任务类型")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["twitter2015", "twitter2017", "200"],
                       help="数据集")
    parser.add_argument("--strategy", type=str, required=True,
                       choices=["none", "ewc", "replay", "lwf", "si", "mas", "gem", "mymethod", "tamcl", "moe"],
                       help="持续学习策略")
    parser.add_argument("--mode", type=str, default="multimodal", choices=["text_only", "multimodal"],
                       help="训练模式")
    parser.add_argument("--use_label_embedding", action="store_true",
                       help="使用标签嵌入")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径")
    
    args = parser.parse_args()
    
    # 创建配置模板
    config = ConfigTemplate()
    
    # 生成脚本
    script_content = config.generate_script(
        args.env, args.task_type, args.dataset, args.strategy, 
        args.mode, args.use_label_embedding
    )
    
    # 生成文件名
    script_name = config.generate_script_name(
        args.env, args.task_type, args.dataset, args.strategy, args.mode
    )
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        output_path = f"scripts/{script_name}"
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(output_path, 0o755)
    
    print(f"脚本已生成: {output_path}")
    print(f"使用方法: ./{output_path}")


if __name__ == "__main__":
    main() 