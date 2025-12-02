"""
模块：checkpoint推理评估工具

使用训练阶段生成的模型/头文件，直接在指定数据集 split 上计算指标。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from types import SimpleNamespace

import torch

from modules.train_utils import create_model
from modules.evaluate import evaluate_single_task
from utils.logger import setup_logger


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_args(task_config: Dict[str, Any], global_params: Dict[str, Any], pretrained_model_path: Optional[str]) -> argparse.Namespace:
    args = argparse.Namespace()
    args.task_name = task_config["task_name"]
    args.session_name = task_config["session_name"]
    args.head_key = task_config.get("head_key", task_config["session_name"])
    args.task_config_file = global_params.get("task_config_file", "")
    args.train_info_json = global_params.get("train_info_json", "checkpoints/train_info.json")
    args.output_model_path = task_config.get("output_model_path", global_params.get("output_model_path", "checkpoints/tmp.pt"))
    args.pretrained_model_path = pretrained_model_path or args.output_model_path

    # 数据参数
    args.data_dir = global_params.get("data_dir", "data")
    args.dataset_name = global_params.get("dataset_name", "twitter2015")
    args.train_text_file = task_config["train_text_file"]
    args.test_text_file = task_config["test_text_file"]
    args.dev_text_file = task_config["dev_text_file"]
    args.image_dir = task_config["image_dir"]

    # 模型参数
    args.text_model_name = task_config["text_model_name"]
    args.image_model_name = task_config["image_model_name"]
    args.fusion_strategy = task_config["fusion_strategy"]
    args.num_heads = task_config["num_heads"]
    args.mode = task_config["mode"]
    args.hidden_dim = task_config["hidden_dim"]
    args.dropout_prob = task_config["dropout_prob"]
    args.num_labels = task_config["num_labels"]

    # 训练/调度参数（为兼容模型创建，值可随意但需存在）
    args.epochs = task_config.get("epochs", 1)
    args.batch_size = task_config.get("batch_size", 8)
    args.lr = task_config.get("lr", 5e-5)
    args.lstm_lr = task_config.get("lstm_lr", 1e-4)
    args.crf_lr = task_config.get("crf_lr", 1e-3)
    args.weight_decay = task_config.get("weight_decay", 0.0)
    args.step_size = task_config.get("step_size", 0)
    args.gamma = task_config.get("gamma", 0.5)
    args.patience = task_config.get("patience", 5)
    args.num_workers = global_params.get("num_workers", 0)
    args.lr_scheduler = task_config.get("lr_scheduler", global_params.get("lr_scheduler", "linear"))
    args.warmup_ratio = task_config.get("warmup_ratio", global_params.get("warmup_ratio", 0.1))

    # 其余标志
    args.use_label_embedding = task_config.get("use_label_embedding", False)
    args.use_hierarchical_head = task_config.get("use_hierarchical_head", False)
    args.label_emb_dim = task_config.get("label_emb_dim", 128)
    args.use_similarity_reg = task_config.get("use_similarity_reg", True)
    args.similarity_weight = task_config.get("similarity_weight", 0.1)
    args.label_embedding_path = task_config.get("label_embedding_path")
    args.triaffine = task_config.get("triaffine", 0)
    args.span_hidden = task_config.get("span_hidden", 256)
    args.use_crf = int(task_config.get("use_crf", 1))
    args.use_span_loss = int(task_config.get("use_span_loss", 0))
    args.boundary_weight = task_config.get("boundary_weight", 0.2)
    args.span_f1_weight = task_config.get("span_f1_weight", 0.0)
    args.transition_weight = task_config.get("transition_weight", 0.0)
    args.graph_smooth = task_config.get("graph_smooth", 0)
    args.graph_tau = task_config.get("graph_tau", 0.5)
    args.use_bilstm = task_config.get("use_bilstm", 0)
    args.bilstm_hidden_size = task_config.get("bilstm_hidden_size", 256)
    args.bilstm_num_layers = task_config.get("bilstm_num_layers", 2)

    return args


def evaluate_checkpoint(config_path: str, session_name: Optional[str] = None, split: str = "dev",
                        model_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    使用现有 checkpoint 在指定 split 上评估。

    Args:
        config_path: 训练配置文件路径
        session_name: 指定评估的 session（None 则评估所有任务）
        split: "dev" 或 "test"
        model_path: 可覆盖配置中的 output_model_path

    Returns:
        每个任务的评估指标列表
    """
    config_path = Path(config_path)
    config = _load_config(config_path)
    tasks = config["tasks"]
    global_params = config.get("global_params", {})
    global_params["task_config_file"] = str(config_path)

    if session_name:
        tasks = [t for t in tasks if t["session_name"] == session_name]
        if not tasks:
            raise ValueError(f"Session {session_name} not found in config {config_path}")

    logger = setup_logger(args=SimpleNamespace(session_name="inference", task_name="inference"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for task in tasks:
        args = _build_args(task, global_params, model_path)
        model = create_model(args, device, label_embedding_manager=None, logger=logger)

        if not args.pretrained_model_path or not Path(args.pretrained_model_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.pretrained_model_path}")

        metrics = evaluate_single_task(model, args.task_name, split, device, args)
        results.append({
            "session_name": args.session_name,
            "task_name": args.task_name,
            "split": split,
            "checkpoint": args.pretrained_model_path,
            "metrics": metrics
        })

    return results
