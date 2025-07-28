#!/usr/bin/env python3
"""
任务配置文件生成器

生成包含所有任务信息的JSON配置文件，用于持续学习训练。
这样可以在训练第i个任务时，预先知道后续任务的信息，实现0样本检测。
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


class TaskConfigGenerator:
    """任务配置生成器"""
    
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
                }
            },
            "twitter2017": {
                "data_dir": "./data",
                "dataset_name": "twitter2017",
                "full_files": {
                    "train": "data/MASC/twitter2017/train.txt",
                    "dev": "data/MASC/twitter2017/dev.txt",
                    "test": "data/MASC/twitter2017/test.txt"
                }
            },
            "mix": {
                "data_dir": "./data",
                "dataset_name": "mix",
                "full_files": {
                    "train": "data/MASC/mix/train.txt",
                    "dev": "data/MASC/mix/dev.txt",
                    "test": "data/MASC/mix/test.txt"
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
        
        # MNER专用数据集配置
        self.mner_datasets = {
            "twitter2015_ner": {
                "data_dir": "./data",
                "dataset_name": "twitter2015",
                "full_files": {
                    "train": "data/MNER/twitter2015/train.txt",
                    "dev": "data/MNER/twitter2015/dev.txt", 
                    "test": "data/MNER/twitter2015/test.txt"
                }
            },
            "twitter2017_ner": {
                "data_dir": "./data",
                "dataset_name": "twitter2017",
                "full_files": {
                    "train": "data/MNER/twitter2017/train.txt",
                    "dev": "data/MNER/twitter2017/dev.txt",
                    "test": "data/MNER/twitter2017/test.txt"
                }
            },
            "mix_ner": {
                "data_dir": "./data",
                "dataset_name": "mix",
                "full_files": {
                    "train": "data/MNER/mix/train.txt",
                    "dev": "data/MNER/mix/dev.txt",
                    "test": "data/MNER/mix/test.txt"
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
            "tam_cl": {
                "params": {
                    "tam_cl": 1
                },
                "description": "TAM-CL"
            },
            "moe": {
                "params": {
                    "moe_adapters": 1,
                    "moe_num_experts": 4,
                    "moe_top_k": 2,
                    "ddas": 1
                },
                "description": "MoE Adapters"
            },
            "clap4clip": {
                "params": {
                    "clap4clip": 1,
                    "adapter_size": 64,
                    "finetune_lambda": 0.1,
                    "temperature": 0.07
                },
                "description": "CLAP4CLIP with Adapters and Probabilistic Finetuning"
            }
        }
    
    def get_dataset_files(self, task_name, dataset):
        # MNER 任务单独处理
        if task_name == "mner":
            if dataset == "200":
                return self.mner_datasets["200_ner"]["files"]
            elif dataset == "twitter2015":
                return self.mner_datasets["twitter2015_ner"]["full_files"]
            elif dataset == "twitter2017":
                return self.mner_datasets["twitter2017_ner"]["full_files"]
            elif dataset == "mix":
                return self.mner_datasets["mix_ner"]["full_files"]
            else:
                raise ValueError(f"Unknown dataset for mner: {dataset}")
        # 其它任务
        else:
            if dataset == "200":
                return self.datasets["200"]["files"]
            elif dataset in ["twitter2015", "twitter2017", "mix"]:
                return self.datasets[dataset]["full_files"]
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
    
    def determine_mode_suffix(self, modes: List[str]) -> str:
        """根据模式列表确定文件后缀"""
        if not modes:
            return ""
        
        unique_modes = set(modes)
        if len(unique_modes) == 1:
            if "text_only" in unique_modes:
                return "t"
            elif "multimodal" in unique_modes:
                return "m"
            else:
                return ""
        else:
            # 既有text_only又有multimodal
            if "text_only" in unique_modes and "multimodal" in unique_modes:
                return "t2m"
            else:
                return ""
    
    def generate_file_names(self, env: str, dataset: str, strategy: str, modes: List[str], 
                           use_label_embedding: bool = False, seq_suffix: str = "") -> Dict[str, str]:
        """生成文件名称"""
        env_config = self.environments[env]
        
        # 确定模式后缀
        mode_suffix = self.determine_mode_suffix(modes)
        
        # 基础名称
        base_name = f"{dataset}_{strategy}"
        if mode_suffix:
            base_name += f"_{mode_suffix}"
        if use_label_embedding:
            base_name += "_label_emb"
        if seq_suffix:
            base_name += f"_{seq_suffix}"
        
        # 模型名称
        if env == "server":
            model_name = "1.pt"
        else:
            model_name = f"model_{base_name}.pt"
        
        # 训练信息JSON
        train_info_json = f"train_info_{base_name}.json"
        
        # 任务头文件
        task_heads_name = f"model_{base_name}_task_heads.pt"
        
        # 标签嵌入文件
        label_embedding_name = f"label_embedding_{base_name}.pt"
        
        return {
            "model_name": model_name,
            "train_info_json": train_info_json,
            "task_heads_name": task_heads_name,
            "label_embedding_name": label_embedding_name,
            "base_name": base_name,
            "mode_suffix": mode_suffix
        }
    
    def create_task_config(self, task_name: str, session_name: str, dataset: str, 
                          env: str, strategy: str, mode: str, **kwargs) -> Dict[str, Any]:
        """创建单个任务的配置"""
        
        # 获取数据集文件
        dataset_files = self.get_dataset_files(task_name, dataset)
        
        # 任务特定参数
        task_specific_params = {
            "masc": {"num_labels": 3, "epochs": 5, "lr": 1e-5, "step_size": 2, "gamma": 0.1},
            "mate": {"num_labels": 3, "epochs": 20, "lr": 5e-5, "step_size": 10, "gamma": 0.5},
            "mner": {"num_labels": 9, "epochs": 20, "lr": 5e-5, "step_size": 10, "gamma": 0.5},
            "mabsa": {"num_labels": 7, "epochs": 20, "lr": 5e-5, "step_size": 10, "gamma": 0.5}
        }
        
        # 如果是200数据集，所有任务的epoch都改为1
        if dataset == "200":
            for task_key in task_specific_params:
                task_specific_params[task_key]["epochs"] = 1
        
        # 合并参数
        task_params = {**task_specific_params.get(task_name, {}), **kwargs}
        
        # 策略参数
        strategy_params = self.strategies.get(strategy, {}).get("params", {})
        
        # 环境配置
        env_config = self.environments[env]

        use_label_embedding = task_params.get("use_label_embedding", False) or kwargs.get("use_label_embedding", False)
        
        if use_label_embedding:
            task_params["lr"] = 1e-3
            task_params["weight_decay"] = 1e-4
            task_params["gamma"] = 0.7
            # 当使用label_embedding时，自动启用混合头
            task_params["use_hierarchical_head"] = True
            
        base_config = {
            "task_name": task_name,
            "session_name": session_name,
            "dataset": dataset,
            "env": env,
            "strategy": strategy,
            "mode": mode,
            # 模型参数
            "num_labels": task_params.get("num_labels", 3),
            "epochs": task_params.get("epochs", 20),
            "lr": task_params.get("lr", 5e-5),
            "batch_size": task_params.get("batch_size", 8),
            "step_size": task_params.get("step_size", 10),
            "gamma": task_params.get("gamma", 0.5),
            "weight_decay": task_params.get("weight_decay", 1e-5),
            "dropout_prob": task_params.get("dropout_prob", 0.1),
            "patience": task_params.get("patience", 5),
            "fusion_strategy": task_params.get("fusion_strategy", "concat"),
            "num_heads": task_params.get("num_heads", 8),
            "hidden_dim": task_params.get("hidden_dim", 768),
            "text_model_name": task_params.get("text_model_name", "microsoft/deberta-v3-base" if strategy != "clap4clip" else "openai/clip-vit-base-patch32"),
            "image_model_name": task_params.get("image_model_name", "google/vit-base-patch16-224-in21k" if strategy != "clap4clip" else "openai/clip-vit-base-patch32"),
            "image_dir": task_params.get("image_dir", "data/img"),
            # 数据集文件
            "train_text_file": dataset_files["train"],
            "test_text_file": dataset_files["test"],
            "dev_text_file": dataset_files["dev"],
            # 持续学习策略参数
            **strategy_params,
            # 标签嵌入
            "use_label_embedding": task_params.get("use_label_embedding", False),
            "use_hierarchical_head": task_params.get("use_hierarchical_head", False),
            "label_emb_dim": task_params.get("label_emb_dim", 128),
            "use_similarity_reg": task_params.get("use_similarity_reg", True),
            "similarity_weight": task_params.get("similarity_weight", 0.1),
            # 模型头部参数
            "triaffine": task_params.get("triaffine", 1),
            "span_hidden": task_params.get("span_hidden", 256),
            # 图平滑参数
            "graph_smooth": task_params.get("graph_smooth", 1),
            "graph_tau": task_params.get("graph_tau", 0.5),
            # 其他参数
            "num_workers": task_params.get("num_workers", 4),
        }
        
        return base_config
    
    def generate_task_sequence_config(self, env: str, dataset: str, 
                                    task_sequence: List[str] = None,
                                    mode_sequence: List[str] = None,
                                    strategy: str = "none",
                                    use_label_embedding: bool = False,
                                    seq_suffix: str = "",
                                    **kwargs) -> Dict[str, Any]:
        """生成完整的任务序列配置"""
        
        if task_sequence is None:
            task_sequence = ["masc", "mate", "mner", "mabsa"]
        
        if mode_sequence is None:
            # 默认所有任务都使用multimodal模式
            mode_sequence = ["multimodal"] * len(task_sequence)
        
        # 确保任务序列和模式序列长度一致
        if len(task_sequence) != len(mode_sequence):
            raise ValueError(f"任务序列长度({len(task_sequence)})与模式序列长度({len(mode_sequence)})不匹配")
        
        # 环境配置
        env_config = self.environments[env]
        
        # 文件名称（基于所有模式）
        file_names = self.generate_file_names(env, dataset, strategy, mode_sequence, use_label_embedding, seq_suffix)
        
        # 创建任务配置列表
        tasks = []
        for i, (task_name, mode) in enumerate(zip(task_sequence, mode_sequence)):
            session_name = f"{task_name}_{i+1}"
            
            # 创建任务配置
            task_config = self.create_task_config(
                task_name=task_name,
                session_name=session_name,
                dataset=dataset,
                env=env,
                strategy=strategy,
                mode=mode,
                use_label_embedding=use_label_embedding,
                **kwargs
            )
            
            # 设置标签嵌入路径
            if task_config.get("use_label_embedding", False):
                task_config["label_embedding_path"] = f"{env_config['checkpoint_dir']}/{file_names['label_embedding_name']}"
            
            tasks.append(task_config)
        
        # 创建完整配置
        config = {
            "env": env,
            "dataset": dataset,
            "strategy": strategy,
            "mode_sequence": mode_sequence,
            "mode_suffix": file_names["mode_suffix"],
            "use_label_embedding": use_label_embedding,
            "seq_suffix": seq_suffix,
            "total_tasks": len(tasks),
            "tasks": tasks,
            "global_params": {
                "base_dir": env_config["base_dir"],
                "output_model_path": f"{env_config['checkpoint_dir']}/{file_names['model_name']}",
                "train_info_json": f"{env_config['checkpoint_dir']}/{file_names['train_info_json']}",
                "task_heads_path": f"{env_config['checkpoint_dir']}/{file_names['task_heads_name']}",
                "label_embedding_path": f"{env_config['checkpoint_dir']}/{file_names['label_embedding_name']}",
                "ewc_dir": env_config["ewc_dir"],
                "gem_mem_dir": env_config["gem_dir"],
                "log_dir": env_config["log_dir"],
                "checkpoint_dir": env_config["checkpoint_dir"],
                "num_workers": 4,
                "data_dir": "./data",
                "dataset_name": dataset,
            }
        }
        
        return config


def main():
    parser = argparse.ArgumentParser(description="生成任务配置文件")
    parser.add_argument("--env", type=str, default="local", 
                       choices=["local", "server"],
                       help="环境类型")
    parser.add_argument("--dataset", type=str, default="200", 
                       choices=["twitter2015", "twitter2017", "mix", "200"],
                       help="数据集名称")
    parser.add_argument("--strategy", type=str, default="none",
                       choices=["none", "ewc", "replay", "lwf", "si", "mas", "gem", "mymethod", "tam_cl", "moe", "clap4clip"],
                       help="持续学习策略")
    parser.add_argument("--task_sequence", type=str, nargs="+", 
                       default=["masc", "mate", "mner", "mabsa", "masc", "mate", "mner", "mabsa"],
                       help="任务序列")
    parser.add_argument("--mode_sequence", type=str, nargs="+", 
                       default=["text_only", "text_only", "text_only", "text_only", "multimodal", "multimodal", "multimodal", "multimodal"],
                       help="模式序列（与任务序列一一对应）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出配置文件路径（可选，会自动生成）")
    parser.add_argument("--use_label_embedding", action="store_true",
                       help="是否使用标签嵌入")
    parser.add_argument("--seq_suffix", type=str, default="",
                       help="序列后缀（如seq1、seq2等）")

    
    args = parser.parse_args()
    
    # 创建配置生成器
    generator = TaskConfigGenerator()
    
    # 生成配置
    config = generator.generate_task_sequence_config(
        env=args.env,
        dataset=args.dataset,
        task_sequence=args.task_sequence,
        mode_sequence=args.mode_sequence,
        strategy=args.strategy,
        use_label_embedding=args.use_label_embedding,
        seq_suffix=args.seq_suffix
    )
    
    # 自动生成文件名
    if args.output is None:
        # 创建configs目录
        configs_dir = Path("scripts/configs")
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        filename_parts = [args.env, args.dataset, args.strategy]
        if args.use_label_embedding:
            filename_parts.append("label_emb")
        filename = "_".join(filename_parts) + ".json"
        
        output_path = configs_dir / filename
    else:
        output_path = Path(args.output)
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"任务配置文件已生成: {output_path}")
    print(f"环境: {args.env}")
    print(f"数据集: {args.dataset}")
    print(f"策略: {args.strategy}")
    print(f"模式后缀: {config['mode_suffix']}")
    print(f"标签嵌入: {'是' if args.use_label_embedding else '否'}")
    print(f"包含 {len(config['tasks'])} 个任务:")
    
    for i, (task, mode) in enumerate(zip(config['tasks'], config['mode_sequence'])):
        print(f"  {i+1}. {task['task_name']} ({task['session_name']}) - {mode}")
        print(f"     数据集: {task['train_text_file']}")
        print(f"     标签数: {task['num_labels']}")
        print(f"     训练轮数: {task['epochs']}")
        print(f"     学习率: {task['lr']}")
        print(f"     批次大小: {task['batch_size']}")
        print()
    
    # 显示文件路径
    print(f"\n文件路径:")
    model_path = config['global_params']['output_model_path'].replace('\\', '/')
    train_info_path = config['global_params']['train_info_json'].replace('\\', '/')
    task_heads_path = config['global_params']['task_heads_path'].replace('\\', '/')
    print(f"  模型文件: {model_path}")
    print(f"  训练信息: {train_info_path}")
    print(f"  任务头文件: {task_heads_path}")
    if args.use_label_embedding:
        label_emb_path = config['global_params']['label_embedding_path'].replace('\\', '/')
        print(f"  标签嵌入: {label_emb_path}")
    
    # 显示使用示例
    print(f"\n使用示例:")
    # 将路径中的反斜杠转换为正斜杠，确保跨平台兼容
    config_path = str(output_path).replace('\\', '/')
    print(f"python -m scripts.train_with_zero_shot --config {config_path}")


if __name__ == "__main__":
    main() 