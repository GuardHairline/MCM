#!/usr/bin/env python3
"""
更细粒度的 checkpoint 对比脚本：
- 支持剥离常见前缀（base_model., model., module.），方便跨框架比对
- 输出各模块参数量（text_encoder / image_encoder / bilstm / classifier / crf / head 等）
- 展示仅在一侧出现的键及其顶层前缀分布
- 列出同名张量的形状差异

用法示例：
    python tools/compare_checkpoints_detailed.py \
        --a checkpoints/bilstm_test_results_account_2\\(12\\)/kaggle_mner_twitter2015_textonly_config_default.pt \
        --b checkpoints/ner_experiments/best_model_exp1.pt
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable
import torch


COMMON_PREFIXES = ["base_model.", "model.", "module."]


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        return {}
    # 常见键
    for k in ["state_dict", "model_state_dict", "model", "net", "params", "module"]:
        if k in obj and isinstance(obj[k], dict):
            inner = obj[k]
            if any(isinstance(v, torch.Tensor) for v in inner.values()):
                return {kk: vv for kk, vv in inner.items() if isinstance(vv, torch.Tensor)}
    # 外层即 state dict
    if any(isinstance(v, torch.Tensor) for v in obj.values()):
        return {kk: vv for kk, vv in obj.items() if isinstance(vv, torch.Tensor)}
    # 兜底：选含 tensor 最多的子 dict
    candidate_dicts = [v for v in obj.values() if isinstance(v, dict)]
    if candidate_dicts:
        best = max(candidate_dicts, key=lambda d: sum(isinstance(v, torch.Tensor) for v in d.values()))
        return {kk: vv for kk, vv in best.items() if isinstance(vv, torch.Tensor)}
    return {}


def strip_prefixes(name: str, prefixes: Iterable[str]) -> str:
    for p in prefixes:
        if name.startswith(p):
            return name[len(p) :]
    return name


def group_stats(sd: Dict[str, torch.Tensor]) -> Counter:
    c = Counter()
    for k, v in sd.items():
        root = k.split(".")[0]
        c[root] += v.numel()
    return c


def print_group(title: str, sd: Dict[str, torch.Tensor]):
    total = sum(v.numel() for v in sd.values())
    print(f"\n=== {title} ===")
    print(f"参数总数: {total/1e6:.2f}M, 条目数: {len(sd)}")
    for tag in ["text_encoder", "image_encoder", "bilstm", "classifier", "crf", "head"]:
        num = sum(v.numel() for k, v in sd.items() if k.startswith(tag))
        if num > 0:
            print(f"  {tag}: {num/1e6:.2f}M")
    top = group_stats(sd).most_common(10)
    print("  顶层前缀 Top-10 (按参数量):")
    for k, n in top:
        print(f"    {k:20s} {n/1e6:6.2f}M")


def prefix_hist(names):
    c = Counter()
    for n in names:
        c[n.split(".")[0]] += 1
    return c.most_common(10)


def main():
    parser = argparse.ArgumentParser(description="Detailed checkpoint comparison")
    parser.add_argument("--a", required=True, help="checkpoint A")
    parser.add_argument("--b", required=True, help="checkpoint B")
    parser.add_argument("--no-strip", action="store_true", help="不剥离常见前缀")
    args = parser.parse_args()

    prefixes = [] if args.no_strip else COMMON_PREFIXES
    sd_a = {strip_prefixes(k, prefixes): v for k, v in load_state_dict(Path(args.a)).items()}
    sd_b = {strip_prefixes(k, prefixes): v for k, v in load_state_dict(Path(args.b)).items()}

    print_group(f"A: {args.a}", sd_a)
    print_group(f"B: {args.b}", sd_b)

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    only_a = list(keys_a - keys_b)
    only_b = list(keys_b - keys_a)
    print(f"\n仅在 A 的键数量: {len(only_a)}")
    print("示例:", only_a[:20])
    print(f"\n仅在 B 的键数量: {len(only_b)}")
    print("示例:", only_b[:20])
    print("\n仅在 A 的顶层前缀 Top-10:", prefix_hist(only_a))
    print("仅在 B 的顶层前缀 Top-10:", prefix_hist(only_b))

    diff_shapes = []
    for k in keys_a & keys_b:
        va, vb = sd_a[k], sd_b[k]
        if va.shape != vb.shape:
            diff_shapes.append((k, tuple(va.shape), tuple(vb.shape)))
    print(f"\n同名但形状不同的层数量: {len(diff_shapes)}")
    print("示例:", diff_shapes[:20])


if __name__ == "__main__":
    main()
