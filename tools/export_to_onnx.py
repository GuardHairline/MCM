#!/usr/bin/env python3
"""
将已训练的 PyTorch checkpoint 导出为 ONNX 模型

用法示例：
    python -m tools.export_to_onnx   --config scripts/configs/kaggle_bilstm_test/account_2/kaggle_bilstm_config_default_twitter2015_mner.json  --session mner_1   --checkpoint checkpoints/bilstm_test_results_account_2(11)/kaggle_mner_twitter2015_tex
tonly_config_default.pt --output onnx/mner_textonly.onnx 
"""

import argparse
import json
from pathlib import Path

import torch

from modules.train_utils import create_model
from utils.logger import setup_logger


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_args(task_cfg: dict, global_params: dict, checkpoint: str):
    # 组装最小的 args 对象，供 create_model 使用
    args = argparse.Namespace()
    args.task_name = task_cfg["task_name"]
    args.session_name = task_cfg["session_name"]
    args.train_info_json = global_params.get("train_info_json", "checkpoints/train_info.json")
    args.output_model_path = task_cfg.get("output_model_path", "checkpoints/tmp.pt")
    args.pretrained_model_path = checkpoint

    # 数据参数
    args.data_dir = global_params.get("data_dir", "data")
    args.dataset_name = global_params.get("dataset_name", "twitter2015")
    args.train_text_file = task_cfg["train_text_file"]
    args.test_text_file = task_cfg["test_text_file"]
    args.dev_text_file = task_cfg["dev_text_file"]
    args.image_dir = task_cfg["image_dir"]

    # 模型参数
    args.text_model_name = task_cfg["text_model_name"]
    args.image_model_name = task_cfg["image_model_name"]
    args.fusion_strategy = task_cfg["fusion_strategy"]
    args.num_heads = task_cfg["num_heads"]
    args.mode = task_cfg["mode"]
    args.hidden_dim = task_cfg["hidden_dim"]
    args.dropout_prob = task_cfg["dropout_prob"]
    args.num_labels = task_cfg["num_labels"]
    args.use_bilstm = task_cfg.get("use_bilstm", 1)
    args.enable_bilstm_head = task_cfg.get("enable_bilstm_head", 1)
    args.bilstm_hidden_size = task_cfg.get("bilstm_hidden_size", 256)
    args.bilstm_num_layers = task_cfg.get("bilstm_num_layers", 2)
    args.use_crf = int(task_cfg.get("use_crf", 1))

    # 其他必要参数（默认）
    args.lr = task_cfg.get("lr", 1e-5)
    args.weight_decay = task_cfg.get("weight_decay", 0.01)
    args.step_size = task_cfg.get("step_size", 0)
    args.gamma = task_cfg.get("gamma", 0.5)
    args.patience = task_cfg.get("patience", 5)
    args.epochs = task_cfg.get("epochs", 1)
    args.batch_size = task_cfg.get("batch_size", 1)
    args.num_workers = task_cfg.get("num_workers", 0)
    args.lr_scheduler = task_cfg.get("lr_scheduler", "linear")
    args.warmup_ratio = task_cfg.get("warmup_ratio", 0.0)

    # 标签嵌入相关
    args.use_label_embedding = task_cfg.get("use_label_embedding", False)
    args.use_hierarchical_head = task_cfg.get("use_hierarchical_head", False)
    args.use_similarity_reg = task_cfg.get("use_similarity_reg", False)
    args.label_emb_dim = task_cfg.get("label_emb_dim", 128)
    args.label_embedding_path = task_cfg.get("label_embedding_path", None)

    # 兼容 DEQA/其他标志
    args.triaffine = task_cfg.get("triaffine", 0)
    args.span_hidden = task_cfg.get("span_hidden", 256)
    args.use_span_loss = task_cfg.get("use_span_loss", 0)
    args.boundary_weight = task_cfg.get("boundary_weight", 0.2)
    args.span_f1_weight = task_cfg.get("span_f1_weight", 0.0)
    args.transition_weight = task_cfg.get("transition_weight", 0.0)
    args.graph_smooth = task_cfg.get("graph_smooth", 0)
    args.graph_tau = task_cfg.get("graph_tau", 0.5)

    # 持续学习/其他标志，供 logger 安全访问（参考 parser.py）
    args.replay = task_cfg.get("replay", 0)
    args.memory_percentage = task_cfg.get("memory_percentage", 0.05)
    args.replay_ratio = task_cfg.get("replay_ratio", 0.5)
    args.replay_frequency = task_cfg.get("replay_frequency", 4)
    args.ewc = task_cfg.get("ewc", 0)
    args.ewc_lambda = task_cfg.get("ewc_lambda", 0.01)
    args.parallel = task_cfg.get("parallel", 0)
    args.lwf = task_cfg.get("lwf", 0)
    args.lwf_T = task_cfg.get("lwf_T", 2.0)
    args.lwf_alpha = task_cfg.get("lwf_alpha", 0.5)
    args.lwf_decay = task_cfg.get("lwf_decay", 0.5)
    args.si = task_cfg.get("si", 0)
    args.si_epsilon = task_cfg.get("si_epsilon", 0.1)
    args.si_decay = task_cfg.get("si_decay", 0.5)
    args.mas = task_cfg.get("mas", 0)
    args.mas_eps = task_cfg.get("mas_eps", 1e-3)
    args.mas_decay = task_cfg.get("mas_decay", 0.5)
    args.gem = task_cfg.get("gem", 0)
    args.gem_mem = task_cfg.get("gem_mem", 100)
    args.gem_dir = task_cfg.get("gem_dir", "gem_memory")
    args.gem_mem_dir = task_cfg.get("gem_mem_dir", "gem_memory")
    args.pnn = task_cfg.get("pnn", 0)
    args.tam_cl = task_cfg.get("tam_cl", 0)
    args.tam_alpha = task_cfg.get("tam_alpha", args.lwf_alpha if hasattr(args, 'lwf_alpha') else 0.5)
    args.clap4clip = task_cfg.get("clap4clip", 0)
    args.deqa = task_cfg.get("deqa", 0)
    args.moe_adapters = task_cfg.get("moe_adapters", 0)
    args.moe_num_experts = task_cfg.get("moe_num_experts", 1)
    args.moe_top_k = task_cfg.get("moe_top_k", 1)
    args.ddas = task_cfg.get("ddas", 0)
    args.ddas_threshold = task_cfg.get("ddas_threshold", 0.0)
    args.moe_balance_coef = task_cfg.get("moe_balance_coef", 0.01)
    args.vis_show_predictions = False
    args.enable_feature_visualization = False
    args.plot_training_curves = False

    return args


def export_onnx(config_path: str, session: str, checkpoint: str, output_path: str):
    config = load_config(Path(config_path))
    tasks = config["tasks"]
    global_params = config.get("global_params", {})
    global_params["task_config_file"] = config_path

    # 选择目标 session
    target = None
    for t in tasks:
        if t["session_name"] == session:
            target = t
            break
    if target is None:
        raise ValueError(f"Session {session} not found in config {config_path}")

    args = build_args(target, global_params, checkpoint)
    logger = setup_logger(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(args, device, label_embedding_manager=None, logger=logger)
    if not Path(args.pretrained_model_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.pretrained_model_path}")
    model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device), strict=False)
    model.eval()

    # 构造 dummy 输入
    batch_size = 1
    seq_len = 16
    input_ids = torch.randint(1, 100, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
    token_type_ids = torch.zeros_like(input_ids).to(device)
    image_tensor = torch.zeros(batch_size, 3, 224, 224).to(device)

    # ONNX 导出
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids, image_tensor),
        output_path.as_posix(),
        input_names=["input_ids", "attention_mask", "token_type_ids", "image_tensor"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "image_tensor": {0: "batch"},
            "logits": {0: "batch", 1: "seq"},
        },
    )
    logger.info(f"ONNX model exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export trained PyTorch model to ONNX")
    parser.add_argument("--config", type=str, required=True, help="Task config path")
    parser.add_argument("--session", type=str, required=True, help="Session name in config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint (.pt) path")
    parser.add_argument("--output", type=str, required=True, help="ONNX output path")
    args = parser.parse_args()

    export_onnx(args.config, args.session, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
