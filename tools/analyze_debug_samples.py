#!/usr/bin/env python3
"""
对比分析 debug_samples.jsonl 文件

用法:
    python tools/analyze_debug_samples.py --file checkpoints/bilstm_test_results_account_2(10)/mner_1_debug_samples.jsonl --top 20
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def span_sets(spans):
    # 过滤掉无效span（如解码错误的负类型）
    return {tuple(s) for s in spans if len(s) == 3 and s[2] >= 0}


def analyze(records):
    stats = {
        "total": len(records),
        "exact_span_match": 0,   # pred spans == gold spans
        "partial_span_overlap": 0,  # 至少有一个span重合
        "token_correct": 0,
        "token_total": 0,
        "pred_only": Counter(),
        "gold_only": Counter(),
        "span_len_mismatch": 0,
        "span_tp": 0,
        "span_fp": 0,
        "span_fn": 0,
    }

    for rec in records:
        pred = span_sets(rec.get("pred_spans", []))
        gold = span_sets(rec.get("gold_spans", []))
        gold_seq = rec.get("gold_seq", [])
        pred_seq = rec.get("pred_seq", [])

        # token-level accuracy（排除O=0和padding=-100）
        token_pairs = [(g, p) for g, p in zip(gold_seq, pred_seq) if g not in (0, -100)]
        stats["token_total"] += len(token_pairs)
        stats["token_correct"] += sum(1 for g, p in token_pairs if g == p)

        if pred == gold:
            stats["exact_span_match"] += 1
        elif pred & gold:
            stats["partial_span_overlap"] += 1

        # 记录预测有而gold没有的span类型
        for s in pred - gold:
            stats["pred_only"][s[2]] += 1
        for s in gold - pred:
            stats["gold_only"][s[2]] += 1

        # span级别TP/FP/FN
        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)
        stats["span_tp"] += tp
        stats["span_fp"] += fp
        stats["span_fn"] += fn

        # 记录长度不一致的case
        if rec.get("valid_len", 0) != max([max(s[1], s[0]) for s in gold], default=0) + 1:
            stats["span_len_mismatch"] += 1

    # 输出概要
    total = stats["total"] or 1
    token_acc = stats["token_correct"] / stats["token_total"] if stats["token_total"] else 0.0
    print(f"总样本: {stats['total']}")
    print(f"所有span完全匹配: {stats['exact_span_match']} ({stats['exact_span_match'] / total * 100:.2f}%)")
    print(f"部分span重合(但不全匹配): {stats['partial_span_overlap']} ({stats['partial_span_overlap'] / total * 100:.2f}%)")
    print(f"token准确率: {token_acc * 100:.2f}% ({stats['token_correct']}/{stats['token_total']})")
    span_precision = stats["span_tp"] / (stats["span_tp"] + stats["span_fp"]) if (stats["span_tp"] + stats["span_fp"]) else 0.0
    span_recall = stats["span_tp"] / (stats["span_tp"] + stats["span_fn"]) if (stats["span_tp"] + stats["span_fn"]) else 0.0
    span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall) if (span_precision + span_recall) else 0.0

    print(f"Span级别 Precision/Recall/F1: {span_precision*100:.2f}% / {span_recall*100:.2f}% / {span_f1*100:.2f}%")
    print(f"预测多出的span类型计数: {dict(stats['pred_only'])}")
    print(f"漏掉的span类型计数: {dict(stats['gold_only'])}")
    print(f"span长度不一致的样本数: {stats['span_len_mismatch']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to debug_samples.jsonl")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    records = load_records(path)
    if not records:
        print("No valid records found.")
        return

    analyze(records)


if __name__ == "__main__":
    main()
