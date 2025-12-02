import torch
import numpy as np
from types import SimpleNamespace

# 1. 导入 Framework 的解码函数
# 假设您在项目根目录下运行
from utils.decode import decode_mner 

# 2. 复制 Simple Script 的解码函数
def extract_entities_simple(labels, label_names=None):
    if label_names is None:
        label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", 
                       "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    entities = []
    current_entity = None
    for i, label_id in enumerate(labels):
        if label_id == -100: continue
        label_name = label_names[label_id] if label_id < len(label_names) else "O"
        if label_name.startswith("B-"):
            if current_entity is not None: entities.append(current_entity)
            entity_type = label_name[2:]
            current_entity = (i, i, entity_type)
        elif label_name.startswith("I-"):
            if current_entity is not None:
                entity_type = label_name[2:]
                if current_entity[2] == entity_type:
                    current_entity = (current_entity[0], i, entity_type)
                else:
                    entities.append(current_entity)
                    current_entity = (i, i, entity_type)
            else:
                entity_type = label_name[2:]
                current_entity = (i, i, entity_type)
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
    if current_entity is not None: entities.append(current_entity)
    return entities

def compute_span_f1_simple(pred_entities, true_entities):
    pred_set = set(pred_entities)
    true_set = set(true_entities)
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

# ==========================================
# 3. 模拟一个典型 Case：Token 部分错误
# ==========================================
print("=== 调试结果对比 ===")

# 模拟：Label (Gold)
# [CLS] B-PER I-PER [SEP] [PAD]
labels_raw = [-100, 1, 2, -100, -100] 

# 模拟：Prediction (Model)
# 模型预测错误：B-PER 被预测对了，但 I-PER 预测成了 O
# [CLS] B-PER O     [SEP] [PAD]  (注：CRF decode 输出通常包含 0)
preds_raw =  [0,    1, 0, 0,    0] 

print(f"Gold: {labels_raw}")
print(f"Pred: {preds_raw}")

# --- Framework 方法 (Filter + decode_mner) ---
valid_mask = [l != -100 for l in labels_raw]
gold_filtered = [l for l, m in zip(labels_raw, valid_mask) if m] # [1, 2]
pred_filtered = [p for p, m in zip(preds_raw, valid_mask) if m]  # [1, 0]

print(f"\n[Framework] Filtered inputs: Gold={gold_filtered}, Pred={pred_filtered}")
fw_gold_chunks = decode_mner(gold_filtered)
fw_pred_chunks = decode_mner(pred_filtered)
print(f"[Framework] Gold Chunks: {fw_gold_chunks}") # {(0, 1, 0)} -> (start, end, type)
print(f"[Framework] Pred Chunks: {fw_pred_chunks}") # {(0, 0, 0)}
# 计算 F1
tp = len(fw_gold_chunks & fw_pred_chunks)
print(f"[Framework] Match (TP): {tp}") # 应该是 0 (因为 (0,1) != (0,0))

# --- Simple Script 方法 (extract_entities) ---
# Simple Script 不做 filter，直接传 list
print(f"\n[Simple] Inputs: Gold={labels_raw}, Pred={preds_raw}")
simple_gold_ents = extract_entities_simple(labels_raw)
simple_pred_ents = extract_entities_simple(preds_raw)
print(f"[Simple] Gold Entities: {simple_gold_ents}") # (1, 2, 'PER')
print(f"[Simple] Pred Entities: {simple_pred_ents}") # (1, 1, 'PER')

simple_f1 = compute_span_f1_simple(simple_pred_ents, simple_gold_ents)
print(f"[Simple] F1: {simple_f1}") # 应该是 0

# ==========================================
# 4. 模拟 Simple Script 可能出问题的地方：索引偏移
# ==========================================
# 假设 labels 和 preds 的长度不一致，或者 attention mask 处理导致 preds 全是 0
# 比如 padding 处，labels 是 -100，preds 是 0 ('O')
# extract_entities 处理 labels 时跳过 -100，处理 preds 时看到 0 认为是 O。这是对的。

# 唯一的可能是 label_names 映射不一致
# 您的 dataset 中 -1 -> 0 (B=1, I=2).
# Simple Script 硬编码: 1=B-PER, 2=I-PER.
# 这也是对的。