# scripts/evaluate.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from datasets.get_dataset import get_dataset
def evaluate_single_task(full_model, task_name, split, device, args):
    """
    对指定任务的 {split} (dev/test) 数据集进行评估，返回准确率(%)。
    """
    ds = get_dataset(task_name, split, args)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    full_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"]
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            image_tensor = batch["image_tensor"].to(device)
            labels = batch["label"].to(device)

            logits = full_model(input_ids, attention_mask, token_type_ids, image_tensor)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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


def evaluate_all_learned_tasks(full_model, task_list, device, args):
    """
    对当前模型在 'task_list' 里所有任务的 test 数据集进行评估，返回 [acc_task1, acc_task2, ...].
    """
    acc_list = []
    for tname in task_list:
        metrics_dict = evaluate_single_task(full_model, tname, "test", device, args)
        acc_list.append(metrics_dict["accuracy"])
    return acc_list


