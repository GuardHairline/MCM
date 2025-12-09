# scripts/generate_ta_pecl_configs.py
import json
import os
import shutil

def generate_ta_pecl_configs():
    # 基础配置
    base_config = {
        "global_params": {
            "train_info_json": "checkpoints/train_info_ta_pecl.json",
            "ewc_dir": "checkpoints/ewc_params",
            "gem_dir": "checkpoints/gem_memory",
            "label_embedding_path": "checkpoints/label_embedding_ta_pecl.pt",
            "log_file": "logs/train_ta_pecl.log"
        },
        "tasks": []
    }

    # 任务序列 (Sequence 1: MASC -> MATE -> MNER -> MABSA)
    # 也可以根据 AGENTS.md 里的 Sequence 1 调整
    task_sequence = [
        {
            "task_name": "masc",
            "dataset_name": "twitter2015",
            "mode": "multimodal",
            "num_labels": 3,
            "epochs": 10, # 根据需要调整
            "batch_size": 16
        },
        {
            "task_name": "mate",
            "dataset_name": "twitter2015",
            "mode": "multimodal",
            "num_labels": 3,
            "epochs": 15,
            "batch_size": 16
        },
        {
            "task_name": "mner",
            "dataset_name": "twitter2015",
            "mode": "multimodal",
            "num_labels": 9,
            "epochs": 15,
            "batch_size": 16
        },
        {
            "task_name": "mabsa",
            "dataset_name": "twitter2015",
            "mode": "multimodal",
            "num_labels": 7,
            "epochs": 15,
            "batch_size": 16
        }
    ]

    # TA-PECL 特定参数
    ta_pecl_params = {
        "ta_pecl": 1,
        "ta_pecl_top_k": 4,
        # 其他基础参数
        "lr": 2e-5,
        "dropout_prob": 0.1,
        "use_label_embedding": False,
        "fusion_strategy": "concat",
        "description_file": "data/descriptions/all_descriptions.jsonl", # 假设你有这个文件用于 expert_deqa
        "use_crf": 1
    }

    # 生成配置列表
    configs = []
    
    # 为每个任务生成一个配置项
    for i, task in enumerate(task_sequence):
        task_config = task.copy()
        
        # 合并 TA-PECL 参数
        task_config.update(ta_pecl_params)
        
        # 构造 Session Name
        session_name = f"{i}_{task['task_name']}_{task['dataset_name']}_ta_pecl"
        task_config["session_name"] = session_name
        
        # 构造数据路径 (假设数据在 data/ 目录下)
        data_base = f"data/{task['task_name'].upper()}/{task['dataset_name']}"
        task_config["data_dir"] = "data" # Parser 可能需要这个
        task_config["train_text_file"] = f"{data_base}/train.txt"
        task_config["dev_text_file"] = f"{data_base}/dev.txt"
        task_config["test_text_file"] = f"{data_base}/test.txt"
        task_config["image_dir"] = "data/img"
        
        # 输出模型路径
        task_config["output_model_path"] = f"checkpoints/{session_name}.pt"
        
        # 如果是第一个任务，不需要加载预训练模型（指 CL 的前一个 checkpoint）
        # 如果是后续任务，加载前一个任务的模型
        if i > 0:
            prev_session = f"{i-1}_{task_sequence[i-1]['task_name']}_{task_sequence[i-1]['dataset_name']}_ta_pecl"
            task_config["pretrained_model_path"] = f"checkpoints/{prev_session}.pt"
        else:
            task_config["pretrained_model_path"] = "" # 从头开始 (加载 HuggingFace 预训练)

        base_config["tasks"].append(task_config)

    # 保存配置
    output_dir = "scripts/configs/ta_pecl"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 保存总的 all_tasks 配置文件
    output_path = os.path.join(output_dir, "ta_pecl_sequence.json")
    with open(output_path, "w") as f:
        json.dump(base_config, f, indent=4)
    print(f"Generated full sequence config: {output_path}")

    # 2. (可选) 保存单独的每个任务的配置文件，方便单独调试
    for task in base_config["tasks"]:
        single_config = {"global_params": base_config["global_params"], "tasks": [task]}
        fname = f"{task['session_name']}.json"
        with open(os.path.join(output_dir, fname), "w") as f:
            json.dump(single_config, f, indent=4)
        print(f"Generated single config: {fname}")

if __name__ == "__main__":
    generate_ta_pecl_configs()