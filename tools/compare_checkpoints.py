#!/usr/bin/env python3
"""
比较两个 PyTorch checkpoint 的结构和参数量，用于定位模型差异。

用法：
    python tools/compare_checkpoints.py \
        --a checkpoints/bilstm_test_results_account_2\\(12\\)/kaggle_mner_twitter2015_textonly_config_default.pt \
        --b checkpoints/ner_experiments/best_model_exp1.pt
"""

import argparse
from pathlib import Path
import torch


def load_state_dict(path: Path):
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        return obj

    # 优先常见键
    for k in ["state_dict", "model_state_dict", "model", "net", "params", "module"]:
        if k in obj and isinstance(obj[k], dict):
            inner = obj[k]
            if any(isinstance(v, torch.Tensor) for v in inner.values()):
                return inner

    # 如果最外层就是 state_dict
    if any(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj

    # 兜底：挑出包含最多 tensor 的内层 dict
    candidate_dicts = [v for v in obj.values() if isinstance(v, dict)]
    if candidate_dicts:
        best = max(candidate_dicts, key=lambda d: sum(isinstance(v, torch.Tensor) for v in d.values()))
        if any(isinstance(v, torch.Tensor) for v in best.values()):
            return best

    return obj


def count_prefix(sd, prefix: str):
    return sum(v.numel() for k, v in sd.items() if k.startswith(prefix) and isinstance(v, torch.Tensor))


def main():
    parser = argparse.ArgumentParser(description="Compare two checkpoints")
    parser.add_argument("--a", required=True, help="Path to checkpoint A")
    parser.add_argument("--b", required=True, help="Path to checkpoint B")
    args = parser.parse_args()

    path_a = Path(args.a)
    path_b = Path(args.b)
    sd_a_raw = load_state_dict(path_a)
    sd_b_raw = load_state_dict(path_b)
    # 只保留张量参数
    sd_a = {k: v for k, v in sd_a_raw.items() if isinstance(v, torch.Tensor)}
    sd_b = {k: v for k, v in sd_b_raw.items() if isinstance(v, torch.Tensor)}

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    total_a = sum(v.numel() for v in sd_a.values())
    total_b = sum(v.numel() for v in sd_b.values())

    print(f"=== A: {path_a} ===")
    print(f"参数总数: {total_a/1e6:.2f}M, 条目数: {len(sd_a)}")
    for prefix in ["text_encoder", "base_model", "bilstm", "classifier", "crf", "head"]:
        if any(k.startswith(prefix) for k in sd_a):
            print(f"  {prefix} 参数: {count_prefix(sd_a, prefix)/1e6:.2f}M")

    print(f"\n=== B: {path_b} ===")
    print(f"参数总数: {total_b/1e6:.2f}M, 条目数: {len(sd_b)}")
    for prefix in ["text_encoder", "base_model", "bilstm", "classifier", "crf", "head"]:
        if any(k.startswith(prefix) for k in sd_b):
            print(f"  {prefix} 参数: {count_prefix(sd_b, prefix)/1e6:.2f}M")

    only_a = list(keys_a - keys_b)
    only_b = list(keys_b - keys_a)
    print(f"\n仅在 A 的键数量: {len(only_a)}")
    print("示例:", only_a[:20])
    print(f"\n仅在 B 的键数量: {len(only_b)}")
    print("示例:", only_b[:20])

    common = keys_a & keys_b
    diff_shapes = []
    for k in common:
        va, vb = sd_a[k], sd_b[k]
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            if va.shape != vb.shape:
                diff_shapes.append((k, tuple(va.shape), tuple(vb.shape)))
    print(f"\n同名但形状不同的层数量: {len(diff_shapes)}")
    print("示例:", diff_shapes[:20])


if __name__ == "__main__":
    main()
