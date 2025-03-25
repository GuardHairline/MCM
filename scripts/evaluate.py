# scripts/evaluate.py
from collections import Counter

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from datasets.get_dataset import get_dataset
from utils.decode import decode_mate, decode_mner, decode_mabsa
import logging

logger = logging.getLogger("evaluate")
def evaluate_single_task(model, task_name, split, device, args):
    """
    对指定任务的 {split} (dev/test) 数据集进行评估，返回准确率(%)。
    """
    if isinstance(args, dict):
        batch_size = args.get("batch_size")
    else:
        batch_size = args.batch_size
    ds = get_dataset(task_name, split, args)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()

    is_sequence_task = (task_name in ["mate", "mner", "mabsa"])

    # 用于 token-level (或句级) 计算
    all_preds_token = []
    all_labels_token = []

    # 用于 chunk-level 计算
    all_chunks_pred = []
    all_chunks_gold = []

    #
    # # ======== 新增：用于只打印前 n 条 Debug ========
    # debug_print_limit = 20
    # debug_print_count = 0
    #
    # # 获取 tokenizer 如果需要可视化 token
    # # (这里假设 model.base_model.text_encoder 是一个 huggingface AutoModel,
    # #  需要你自己根据项目结构拿到 tokenizer 或在 dataset 里存.
    # #  如果此处无法直接拿到, 也可在 dataset 那边保留 'raw_text' 用于打印)
    # tokenizer = None
    # if hasattr(model.base_model, "text_encoder") and hasattr(model.base_model.text_encoder, "config"):
    #     from transformers import AutoTokenizer
    #     tokenizer_name = model.base_model.text_encoder.name_or_path  # 可能要看看属性是什么
    #     try:
    #         tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    #     except:
    #         tokenizer = None

    label_counter = Counter()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            image_tensor = batch["image_tensor"].to(device)
            labels = batch["labels"].to(device)


            if is_sequence_task:
                # 获取序列特征，形状为 (batch_size, seq_len, fusion_dim)
                fused_feat = model.base_model(
                    input_ids, attention_mask, token_type_ids, image_tensor,
                    return_sequence=True
                )
                # 统计标签数量（排除-100）
                for label in labels.view(-1).cpu().tolist():
                    if label != -100:
                        label_counter[label] += 1

                if task_name == "mate":
                    # mate 使用 CRF，因此 head 返回的是 list[List[int]]
                    preds_list = model.head(fused_feat, labels=None)  # 进行推理，不传 labels
                    # 构造 mask：仅保留有效 token（labels != -100）
                    mask = (labels != -100)
                    batch_preds = []
                    batch_labels = []
                    # 遍历每个序列
                    for i in range(labels.size(0)):
                        # 对于每个样本，取出 mask 有效的 token 位置
                        seq_mask = mask[i]
                        true_labels = labels[i][seq_mask].cpu().tolist()
                        # preds_list[i] 的长度应当与 seq_mask.sum() 相等
                        pred_labels = preds_list[i]
                        batch_preds.append(pred_labels)
                        batch_labels.append(true_labels)
                        # 对当前序列进行 chunk-level 解码
                        pred_chunks = decode_mate(pred_labels)
                        gold_chunks = decode_mate(true_labels)
                        all_chunks_pred.append(pred_chunks)
                        all_chunks_gold.append(gold_chunks)
                        # 将所有样本的 token 预测和真实标签展平后汇总
                    for seq in batch_preds:
                        all_preds_token.extend(seq)
                    for seq in batch_labels:
                        all_labels_token.extend(seq)
                else:
                    logits = model.head(fused_feat)        # logits: [batch_size, seq_len, num_labels]
                    preds = torch.argmax(logits, dim=2)   # preds => [batch_size, seq_len]
                    # 将它们 flatten (含-100) 以计算 token-level 参考指标
                    all_preds_token.extend(preds.view(-1).cpu().tolist())
                    all_labels_token.extend(labels.view(-1).cpu().tolist())

                    # 开始做 chunk-level decode
                    bsz, seqlen = preds.shape
                    for i in range(bsz):
                        # 过滤 -100
                        valid_len = (labels[i] != -100).sum().item() + 1
                        pred_i = preds[i, :valid_len].cpu().tolist()
                        gold_i = labels[i, :valid_len].cpu().tolist()

                        if task_name == "mner":
                            pred_chunks = decode_mner(pred_i)
                            gold_chunks = decode_mner(gold_i)
                        elif task_name == "mabsa":
                            pred_chunks = decode_mabsa(pred_i)
                            gold_chunks = decode_mabsa(gold_i)
                        else:
                            pred_chunks = set()
                            gold_chunks = set()

                        all_chunks_pred.append(pred_chunks)
                        all_chunks_gold.append(gold_chunks)
                        all_preds_token.extend(pred_i)
                        all_labels_token.extend(gold_i)

                    # # ========== Debug Print: 只打印前 debug_print_limit 条 ==========
                    # if debug_print_count < debug_print_limit:
                    #     debug_print_count += 1
                    #     logger.info(f"\n========== [DEBUG] Sample #{debug_print_count}  ==========")
                    #
                    #     # 如果能拿到 tokenizer，就把 input_ids -> tokens
                    #     if tokenizer:
                    #         # 取对应的 input_ids[i,:valid_len]
                    #         input_ids_i = input_ids[i, :valid_len].cpu().tolist()
                    #         tokens = tokenizer.convert_ids_to_tokens(input_ids_i, skip_special_tokens=False)
                    #         # 打印 tokens + gold + pred
                    #         logger.info("Tokens:")
                    #         for tk_idx, tk in enumerate(tokens):
                    #             logger.info(f"  idx={tk_idx}, token={tk}, gold={gold_i[tk_idx]}, pred={pred_i[tk_idx]}")
                    #     else:
                    #         logger.info(f"input_ids: {input_ids[i, :valid_len].cpu().tolist()}")
                    #         logger.info(f"gold: {gold_i}")
                    #         logger.info(f"pred: {pred_i}")
                    #
                    #     logger.info(f"Gold chunks: {gold_chunks}")
                    #     logger.info(f"Pred chunks: {pred_chunks}")
            else:
                # === 句级分类 ===
                label_counter.update(labels.cpu().tolist())
                fused_cls = model.base_model(input_ids, attention_mask, token_type_ids, image_tensor,
                                             return_sequence=False)
                logits = model.head(fused_cls)  # => (b, num_labels)
                preds = torch.argmax(logits, dim=1)
                all_preds_token.extend(preds.cpu().tolist())
                all_labels_token.extend(labels.cpu().tolist())

    # === token-level or 句级 ACC ===
    # 先过滤掉 -100
    valid_preds = []
    valid_labels = []
    for p, g in zip(all_preds_token, all_labels_token):
        if g != -100:
            valid_preds.append(p)
            valid_labels.append(g)
    # token_acc = accuracy_score(valid_labels, valid_preds)
    #
    # if is_sequence_task:
    #     # 计算 chunk-level P/R/F1
    #     tp, fp, fn = 0, 0, 0
    #     for pset, gset in zip(all_chunks_pred, all_chunks_gold):
    #         tp_ = len(pset.intersection(gset))
    #         fp_ = len(pset - gset)
    #         fn_ = len(gset - pset)
    #         tp += tp_
    #         fp += fp_
    #         fn += fn_
    #     prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    #     rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    #     f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    #
    #     metrics = {
    #         "token_acc": token_acc * 100.0,
    #         "chunk_precision": prec * 100.0,
    #         "chunk_recall": rec * 100.0,
    #         "chunk_f1": f1 * 100.0
    #     }
    #     metrics["accuracy"] = metrics["chunk_f1"]  # 或者用token_acc
    #
    # else:
    #     # 句级分类
    #     precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    #         all_labels_token, all_preds_token, average='macro', zero_division=0
    #     )
    #     acc = accuracy_score(all_labels_token, all_preds_token)
    #     metrics = {
    #         "accuracy": acc * 100.0,
    #         "precision_macro": precision_macro * 100.0,
    #         "recall_macro": recall_macro * 100.0,
    #         "f1_macro": f1_macro * 100.0,
    #     }
    # 计算微平均指标（precision, recall, f1）以及准确率
    micro_prec, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        valid_labels, valid_preds, average="micro", zero_division=0
    )
    acc = accuracy_score(valid_labels, valid_preds)

    metrics = {
        "acc": acc * 100.0,
        "micro_prec": micro_prec * 100.0,
        "micro_recall": micro_recall * 100.0,
        "micro_f1": micro_f1 * 100.0,
        "label_counter": label_counter,
    }
    # 在函数结束时打印或返回标签分布
    logger.info(f"Label Distribution for {task_name} {split}: {label_counter}")
    metrics["label_counter"] = label_counter  # 如果需要将统计结果返回
    return metrics


def evaluate_all_learned_tasks(model, sessions_list, device, train_info):
    acc_list = []
    for session in sessions_list:
        tname = session["task_name"]
        args = session["args"]
        metrics  = evaluate_single_task(model, tname, "test", device, args)
        # 你可取 chunk_f1 或 accuracy 作为acc
        # if tname in ["mate", "mner", "mabsa"]:
        #     acc_list.append(m["chunk_f1"])
        # else:
        #     acc_list.append(m["accuracy"])
        logger.info(f"[Info] Evaluation metrics for {tname} on test set: {metrics}")
        acc_list.append(metrics["acc"])
    return acc_list


