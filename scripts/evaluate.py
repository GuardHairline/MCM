# scripts/evaluate.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from datasets.get_dataset import get_dataset
def evaluate_single_task(model, task_name, split, device, args):
    """
    对指定任务的 {split} (dev/test) 数据集进行评估，返回准确率(%)。
    """
    ds = get_dataset(task_name, split, args)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model.eval()
    all_preds = []
    all_labels = []

    is_sequence_task = (task_name in ["mate", "mner", "mabsa"])

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            image_tensor = batch["image_tensor"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids, image_tensor)

            if is_sequence_task:
                # logits: [batch_size, seq_len, num_labels]
                # preds => [batch_size, seq_len]
                preds = torch.argmax(logits, dim=2)

                # flatten & filter掉 -100
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                valid_mask = (labels_np != -100)  # boolean mask

                # 把有效的预测和标签展平
                all_preds.extend(preds_np[valid_mask].tolist())
                all_labels.extend(labels_np[valid_mask].tolist())

            else:
                # 句级分类 => logits: [batch_size, num_labels]
                # preds => [batch_size]
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

    # 计算 Accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # 计算 Precision / Recall / F1
    # average='macro' 表示对每个类分别计算，再平均；'micro' 考虑整体
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )

    return {
        "accuracy": accuracy * 100.0,  # 转成百分比
        "precision_macro": precision_macro * 100.0,
        "recall_macro": recall_macro * 100.0,
        "f1_macro": f1_macro * 100.0,
        "precision_micro": precision_micro * 100.0,
        "recall_micro": recall_micro * 100.0,
        "f1_micro": f1_micro * 100.0
    }


def evaluate_all_learned_tasks(model, task_list, device, args):
    """
    对当前模型在 'task_list' 里所有任务的 test 数据集进行评估，返回 [acc_task1, acc_task2, ...].
    """
    acc_list = []
    for tname in task_list:
        metrics_dict = evaluate_single_task(model, tname, "test", device, args)
        acc_list.append(metrics_dict["accuracy"])
    return acc_list


