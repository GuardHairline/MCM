# scripts/evaluate.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from datasets.get_dataset import get_dataset
from utils.decode import decode_mate, decode_mner, decode_mabsa

def evaluate_single_task(model, task_name, split, device, args):
    """
    对指定任务的 {split} (dev/test) 数据集进行评估，返回准确率(%)。
    """
    ds = get_dataset(task_name, split, args)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model.eval()

    is_sequence_task = (task_name in ["mate", "mner", "mabsa"])

    # 用于 token-level (或句级) 计算
    all_preds_token = []
    all_labels_token = []

    # 用于 chunk-level 计算
    all_chunks_pred = []
    all_chunks_gold = []

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
                fused_feat = model.base_model(input_ids, attention_mask, token_type_ids, image_tensor, return_sequence=True)
                logits = model.head(fused_feat)
                # logits: [batch_size, seq_len, num_labels]
                # preds => [batch_size, seq_len]
                preds = torch.argmax(logits, dim=2)

                # 将它们 flatten (含-100) 以计算 token-level 参考指标
                all_preds_token.extend(preds.view(-1).cpu().tolist())
                all_labels_token.extend(labels.view(-1).cpu().tolist())

                # 开始做 chunk-level decode
                bsz, seqlen = preds.shape
                for i in range(bsz):
                    # 过滤 -100
                    valid_len = (labels[i] != -100).sum().item()
                    pred_i = preds[i, :valid_len].cpu().tolist()
                    gold_i = labels[i, :valid_len].cpu().tolist()

                    if task_name == "mate":
                        pred_chunks = decode_mate(pred_i)
                        gold_chunks = decode_mate(gold_i)
                    elif task_name == "mner":
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

            else:
                # === 句级分类 ===
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
    token_acc = accuracy_score(valid_labels, valid_preds)

    if is_sequence_task:
        # 计算 chunk-level P/R/F1
        tp, fp, fn = 0, 0, 0
        for pset, gset in zip(all_chunks_pred, all_chunks_gold):
            tp_ = len(pset.intersection(gset))
            fp_ = len(pset - gset)
            fn_ = len(gset - pset)
            tp += tp_
            fp += fp_
            fn += fn_
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

        metrics = {
            "token_acc": token_acc * 100.0,
            "chunk_precision": prec * 100.0,
            "chunk_recall": rec * 100.0,
            "chunk_f1": f1 * 100.0
        }
        metrics["accuracy"] = metrics["chunk_f1"]  # 或者用token_acc

    else:
        # 句级分类
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels_token, all_preds_token, average='macro', zero_division=0
        )
        acc = accuracy_score(all_labels_token, all_preds_token)
        metrics = {
            "accuracy": acc * 100.0,
            "precision_macro": precision_macro * 100.0,
            "recall_macro": recall_macro * 100.0,
            "f1_macro": f1_macro * 100.0,
        }

    return metrics


def evaluate_all_learned_tasks(model, task_list, device, args):
    acc_list = []
    for tname in task_list:
        m = evaluate_single_task(model, tname, "test", device, args)
        # 你可取 chunk_f1 或 accuracy 作为acc
        if tname in ["mate", "mner", "mabsa"]:
            acc_list.append(m["chunk_f1"])
        else:
            acc_list.append(m["accuracy"])
    return acc_list


