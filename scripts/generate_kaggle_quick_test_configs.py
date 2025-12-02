#!/usr/bin/env python3
"""
在 Kaggle 上生成快速回归用的配置：
- 每个配置包含 200 样本、epoch=2、batch_size=4
- 任务序列：masc(text)→mate(text)→mner(text)→mabsa(text)→masc(mm)→mate(mm)→mner(mm)→mabsa(mm)
- 支持多种持续学习策略：none/replay/ewc/si/gem/mas/moe_adapters/lwf/tam_cl
生成位置：scripts/configs/kaggle_quick_test/<method>/kaggle_quick_<method>.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


TASK_SEQ = [
    ("masc", "text_only"),
    ("mate", "text_only"),
    ("mner", "text_only"),
    ("mabsa", "text_only"),
    ("masc", "multimodal"),
    ("mate", "multimodal"),
    ("mner", "multimodal"),
    ("mabsa", "multimodal"),
]
HEAD_KEY_MAP = {
    "masc": "masc_shared",
    "mate": "mate_shared",
    "mner": "mner_shared",
    "mabsa": "mabsa_shared",
}

NUM_LABELS = {"masc": 3, "mate": 3, "mner": 9, "mabsa": 7}


def build_tasks(shared_heads: bool = True) -> List[Dict]:
    tasks = []
    for idx, (name, mode) in enumerate(TASK_SEQ, start=1):
        if name == "mner":
            data_dir = "data/MNER/twitter2015"
        else:
            data_dir = "data/MASC/twitter2015"
        tasks.append(
            {
                "task_name": name,
                "session_name": f"{name}_{idx}",
                "head_key": HEAD_KEY_MAP[name] if shared_heads else f"{name}_{idx}",
                "dataset": "200",
                "mode": mode,
                "train_text_file": f"{data_dir}/train__.txt",
                "dev_text_file": f"{data_dir}/dev__.txt",
                "test_text_file": f"{data_dir}/test__.txt",
                "image_dir": "data/img",
                "num_labels": NUM_LABELS[name],
                "text_model_name": "microsoft/deberta-v3-base",
                "image_model_name": "google/vit-base-patch16-224-in21k",
                "fusion_strategy": "concat",
                "hidden_dim": 768,
                "dropout_prob": 0.3,
                "num_heads": 8,
                # 训练参数（最小化）
                "epochs": 2,
                "batch_size": 4,
                "lr": 5e-5,
                "lstm_lr": 1e-4,
                "crf_lr": 1e-3,
                "weight_decay": 1e-5,
                "step_size": 10,
                "gamma": 0.5,
                "patience": 999,
                # 模型头
                "triaffine": 0,
                "span_hidden": 256,
                "use_crf": 1,
                "use_span_loss": 0,
                "boundary_weight": 0.2,
                "span_f1_weight": 0.0,
                "transition_weight": 0.0,
                # 图平滑
                "graph_smooth": 1,
                "graph_tau": 0.5,
                # 其他
                "use_label_embedding": 0,
                "use_hierarchical_head": 0,
                "num_workers": 0,
                "description_file": None,
            }
        )
    return tasks


def method_flags(method: str) -> Dict[str, int]:
    flags = {
        "ewc": 0,
        "fisher_selector": 0,
        "replay": 0,
        "lwf": 0,
        "si": 0,
        "mas": 0,
        "gem": 0,
        "agem": 0,
        "moe_adapters": 0,
        "tam_cl": 0,
        "deqa": 0,
        "clap4clip": 0,
    }
    key_map = {
        "none": None,
        "replay": "replay",
        "ewc": "ewc",
        "si": "si",
        "gem": "gem",
        "mas": "mas",
        "moe_adapters": "moe_adapters",
        "lwf": "lwf",
        "tam_cl": "tam_cl",
    }
    flag = key_map.get(method)
    if flag:
        flags[flag] = 1
    return flags


def build_config(method: str, output_dir: Path, shared_heads: bool = True) -> Dict:
    tasks = build_tasks(shared_heads=shared_heads)
    cfg = {
        "env": "kaggle",
        "dataset": "200",
        "strategy": method,
        "mode_sequence": [m for (_, m) in TASK_SEQ],
        "mode_suffix": "t2m",
        "use_label_embedding": False,
        "seq_suffix": f"kaggle_quick_{method}",
        "total_tasks": len(tasks),
        "tasks": tasks,
        "global_params": {
            "base_dir": "./",
            "output_model_path": f"checkpoints/kaggle_quick/{method}.pt",
            "train_info_json": f"checkpoints/kaggle_quick/train_info_{method}.json",
            "task_heads_path": f"checkpoints/kaggle_quick/{method}_task_heads.pt",
            "label_embedding_path": f"checkpoints/kaggle_quick/label_embedding_{method}.pt",
            "ewc_dir": "checkpoints/kaggle_quick/ewc_params",
            "gem_mem_dir": "checkpoints/kaggle_quick/gem_memory",
            "log_dir": "checkpoints/kaggle_quick/logs",
            "checkpoint_dir": "checkpoints/kaggle_quick",
            "num_workers": 0,
            "data_dir": "./data",
            "dataset_name": "200",
            "debug_samples": 0,
        },
    }
    cfg.update(method_flags(method))
    out_dir = output_dir / method
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"kaggle_quick_{method}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"✓ 生成 {path}")
    return cfg


def generate(methods: List[str], output_dir: Path):
    for m in methods:
        build_config(m, output_dir, shared_heads=True)
    # 额外生成一个 none_8heads 配置（不共享头，每个 session 独立）
    build_config("none_8heads", output_dir, shared_heads=False)
    # 生成运行脚本
    run_sh = output_dir / "run_all.sh"
    with run_sh.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#!/bin/bash\nset -e\n\n")
        f.write('configs=(\n')
        for m in methods:
            f.write(f"  \"scripts/configs/kaggle_quick_test/{m}/kaggle_quick_{m}.json\"\n")
        # 加入独立头版本
        f.write("  \"scripts/configs/kaggle_quick_test/none_8heads/kaggle_quick_none_8heads.json\"\n")
        f.write(")\n\n")
        f.write('for cfg in "${configs[@]}"; do\n')
        f.write('  echo ">>> Running $cfg"\n')
        f.write('  python -m scripts.train_with_zero_shot --config "$cfg"\n')
        f.write("done\n")
    run_sh.chmod(0o755)
    print(f"✓ 生成批量运行脚本: {run_sh}")


def main():
    parser = argparse.ArgumentParser(description="生成 Kaggle 快速回归配置")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["none", "replay", "ewc", "si", "gem", "mas", "moe_adapters", "lwf", "tam_cl"],
        help="持续学习方法列表",
    )
    parser.add_argument(
        "--output_dir",
        default="scripts/configs/kaggle_quick_test",
        help="配置输出目录",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    generate(args.methods, output_dir)


if __name__ == "__main__":
    main()
