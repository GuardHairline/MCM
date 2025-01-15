# scripts/train.py
import os
import argparse
from collections import Counter
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR  # 用于学习率调度(示例)

from datasets.get_dataset import get_dataset
from scripts.evaluate import evaluate_single_task, evaluate_all_learned_tasks
from models.base_model import BaseMultimodalModel
from models.task_heads.get_head import get_head
from continual.ewc import MultiTaskEWC  # 如果需要 EWC
from continual.metrics import ContinualMetrics, compute_metrics_example
from utils.logging import setup_logger
import logging

class Full_Model(nn.Module):
    def __init__(self, base_model, head, dropout_prob=0.1):
        super().__init__()
        self.base_model = base_model
        self.head = head
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, attention_mask, token_type_ids, image_tensor):
        fused_feat = self.base_model(input_ids, attention_mask, token_type_ids, image_tensor)
        # 在融合输出后加入 dropout
        fused_feat = self.dropout(fused_feat)
        logits = self.head(fused_feat)
        return logits


def train(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_list = args.task_list

    # 用于记录持续学习指标
    cm = ContinualMetrics()
    # 如果希望从上次训练的 metrics 恢复
    if args.load_metrics and os.path.exists(args.load_metrics):
        cm.load_from_json(args.load_metrics)
        logger.info(f"Loaded previous metrics from {args.load_metrics}")

    # 记录本次训练过程的信息(任务, 数据, 路径, 最终指标等)
    train_info = {
        "train_session": args.session_name,  # 自定义, 比如 "session_20231008"
        "task_list": task_list,
        "epochs": args.epochs,
        "details": {},  # 每个task的路径或其他信息
        "final_continual_metrics": {},
        "acc_matrix": []
    }

    # 依次学 tasks (也可能做多任务混合训练)
    for k, task_name in enumerate(task_list):
        # 1) 初始化模型 (base + head)
        base_model = BaseMultimodalModel(args.text_model_name, args.image_model_name,
                                         multimodal_fusion=args.fusion_strategy, num_heads=args.num_heads)
        if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
            logger.info(f"Loading pretrained model from {args.pretrained_model_path}")
            base_model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)

        head = get_head(task_name, base_model, args)
        full_model = Full_Model(base_model, head, dropout_prob=args.dropout_prob)
        full_model.to(device)

        # 确保所有参数都启用梯度
        for name, param in full_model.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True

        # == 使用 weight_decay 来做 L2 正则，缓解过拟合 ==
        optimizer = AdamW(full_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # == 学习率调度器(示例: 每隔 step_size=2 个 epoch, lr缩放 gamma=0.1)==
        scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

        # 2) 如果需要 EWC 并且不是第一个任务
        ewc = None
        if args.strategy == 'ewc' :  # 只有第k>0时, 才算是有旧任务
            ewc = MultiTaskEWC(
                model=full_model,
                current_task_name=task_name,
                ewc_lambda=args.ewc_lambda,
                ewc_dir="ewc_params"
            )
            # 加载所有旧任务的fisher
            ewc.load_all_previous_tasks()

        # 3) 准备当前任务训练集
        train_dataset = get_dataset(task_name, "train", args)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Early Stopping 相关
        best_dev_acc = 0.0
        no_improve_count = 0
        patience = args.patience  # max epochs to wait for improvement

        # 记录本任务过程
        info_this_task = {
            "train_loss_each_epoch": [],
            "dev_metrics_each_epoch": [],  # 这里存各 Epoch 的 {acc, prec, recall, f1}
            "final_dev_acc": 0.0,
            "final_test_acc": 0.0
        }

        # 4) 训练
        for epoch in range(args.epochs):
            start_time = time.time()  # 开始计时
            full_model.train()
            total_loss = 0.0
            label_counter = Counter()

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"]
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                image_tensor = batch["image_tensor"].to(device)
                labels = batch["label"].to(device)

                label_counter.update(labels.cpu().numpy())

                optimizer.zero_grad()
                logits = full_model(input_ids, attention_mask, token_type_ids, image_tensor)
                loss = F.cross_entropy(logits, labels)

                if ewc:
                    loss += ewc.penalty(full_model)

                loss.backward()
                # 可以在此处做梯度裁剪(clip)防梯度爆炸
                torch.nn.utils.clip_grad_norm_(full_model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            # 记录一个 epoch 的平均损失
            avg_loss = total_loss / len(train_loader)

            # 调整学习率(如果使用scheduler)
            scheduler.step()

            # 评估 dev
            dev_metrics = evaluate_single_task(full_model, task_name, "dev", device, args)
            info_this_task["train_loss_each_epoch"].append(avg_loss)
            info_this_task["dev_metrics_each_epoch"].append(dev_metrics)

            flag_save = False
            # 早停判断
            if dev_metrics["accuracy"] > best_dev_acc:
                flag_save = True
                best_dev_acc = dev_metrics["accuracy"]
                no_improve_count = 0
                torch.save(full_model.state_dict(), "checkpoints/best_model.pt")
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.info(f"[Early Stopping] Dev accuracy hasn't improved for {patience} epochs.")
                    break
            end_time = time.time()  # 结束计时
            elapsed_time = (end_time - start_time) / 60  # 计算耗时

            logger.info(f"[Task={task_name}] Epoch {epoch + 1}/{args.epochs}, "
                  f"Loss={avg_loss:.4f}, "
                  f"Acc(micro_f1)={dev_metrics['accuracy']:.2f}%, "
                  f"Pre_macro={dev_metrics['precision_macro']:.2f}%, "
                  f"Recall_macro={dev_metrics['recall_macro']:.2f}%, "
                  f"f1_macro={dev_metrics['f1_macro']:.2f}%, "
                  f"LabelDist={label_counter}, "
                  f"Epoch processed in {elapsed_time:.4f} minutes.")

        # == 加载最佳模型(如果使用早停+临时保存) ==
        best_model_path = "checkpoints/best_model.pt"
        if os.path.exists(best_model_path) and flag_save:
            logger.info("[train.py] Loading best val_acc model for final testing ...")
            full_model.load_state_dict(torch.load(best_model_path))

        # 学完第 k 个任务后:
        # - 记录 final dev/test acc for this task
        final_dev_metrics = evaluate_single_task(full_model, task_name, "dev", device, args)
        final_test_metrics = evaluate_single_task(full_model, task_name, "test", device, args)
        info_this_task["final_dev_metrics"] = final_dev_metrics
        info_this_task["final_test_metrics"] = final_test_metrics

        # 训练完后 => 估计Fisher并保存
        if ewc:
            ewc.estimate_and_save_fisher(train_loader, device=device, sample_size=200)

        # 保存最终模型
        torch.save(full_model.state_dict(), args.output_model_path)
        logger.info("Model saved at", args.output_model_path)

        # 对所有已学任务的test集评估 => performance_list
        performance_list = evaluate_all_learned_tasks(full_model, task_list[:k + 1], device, args)
        logger.info(f"[train.py] After finishing task {task_name}, test acc on tasks={performance_list}")
        cm.update_acc_matrix(k, performance_list)

        # 保存本任务信息
        train_info["details"][task_name] = info_this_task

        if args.save_each_task:
            ckpt_path = f"checkpoints/{task_name}_model.pt"
            torch.save(full_model.state_dict(), ckpt_path)
            logger.info(f"[train.py] (Optional) Saved model checkpoint for task={task_name} => {ckpt_path}")

    # 所有任务学完 => 计算多任务指标
    k_final = len(task_list)
    final_metrics = compute_metrics_example(cm, k_final)
    logger.info("[train.py] Final Continual Metrics:", final_metrics)
    # 也可以将 cm.acc_matrix 进行保存
    train_info["final_continual_metrics"] = final_metrics
    train_info["acc_matrix"] = cm.acc_matrix

    # 保存 metrics
    if args.save_metrics:
        cm.save_to_json(args.save_metrics)
        logger.info(f"[train.py] Saved acc_matrix to {args.save_metrics}")

    # 保存 train_info
    if args.train_info_json:
        with open(args.train_info_json, "w", encoding="utf-8") as f:
            json.dump(train_info, f, indent=2)
        logger.info(f"[train.py] Train info saved to {args.train_info_json}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_list", type=str, nargs='+', default="masc",
                        help="List of tasks to train on sequentially")
    parser.add_argument("--train_text_file", type=str, default="data/MASC/twitter_data/twitter2015/train.txt")
    parser.add_argument("--test_text_file", type=str, default="data/MASC/twitter_data/twitter2015/test.txt")
    parser.add_argument("--dev_text_file", type=str, default="data/MASC/twitter_data/twitter2015/dev.txt")
    parser.add_argument("--image_dir", type=str, default="data/MASC/twitter_data/twitter2015_images")
    parser.add_argument("--text_model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--image_model_name", type=str, default="resnet50")
    parser.add_argument("--fusion_strategy", type=str, default="multi_head_attention", choices=["concat", "multi_head_attention", "add"])
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_labels", type=int, default=3)  # -1, 0, 1
    parser.add_argument("--strategy", type=str, default="ewc")  # ewc / none ...
    parser.add_argument("--ewc_lambda", type=float, default=1000)
    parser.add_argument("--output_model_path", type=str, default="checkpoints/model_1.pt")
    parser.add_argument("--pretrained_model_path", type=str, default="",
                        help="Path to a pretrained model to continue training")

    # == 新增正则化和防过拟合的超参 ==
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularization).")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout probability in Full_Model.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping (epochs).")

    parser.add_argument("--save_each_task", action="store_true",
                        help="Whether to save model checkpoint after finishing each task")
    parser.add_argument("--load_metrics", type=str, default="",
                        help="Path to a JSON from which to load existing ContinualMetrics")
    parser.add_argument("--save_metrics", type=str, default="checkpoints/acc_matrix.json",
                        help="Path to a JSON to save updated ContinualMetrics at the end")
    parser.add_argument("--train_info_json", type=str, default="checkpoints/train_info.json",
                        help="Path to record train info (tasks, data, metrics, etc.)")
    parser.add_argument("--session_name", type=str, default="default_session",
                        help="Name or ID for this training session")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = setup_logger(logging.INFO)  # 或者设置其他等级 DEBUG/ERROR 等

    train(args, logger)


if __name__ == "__main__":
    main()
