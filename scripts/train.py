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

from continual.experience_replay import ExperienceReplayMemory, default_replay_condition, make_dynamic_replay_condition
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

    new_task_name = args.task_name

    logger.info(f"=== Start training for new task: {new_task_name} ===")
    # ========== 1) 从 JSON 加载或初始化 train_info ==========
    train_info = {}
    if os.path.exists(args.train_info_json):
        with open(args.train_info_json, "r", encoding="utf-8") as f:
            try:
                train_info = json.load(f)
                logger.info(f"Loaded existing training info from {args.train_info_json}")
            except:
                logger.warning(f"Failed to load {args.train_info_json}, using empty info.")
                train_info = {}
    if "tasks" not in train_info:
        train_info["tasks"] = []
    if "acc_matrix" not in train_info:
        # 用来记录所有任务的 a_{k,j} 矩阵, 这里也可用 ContinualMetrics
        train_info["acc_matrix"] = []
    if "sessions" not in train_info:
        train_info["sessions"] = []

    # 旧任务数量
    old_sessions = train_info["sessions"]  # list[str]
    old_sessions_count = len(old_sessions)
    logger.info(f"Previously learned sessions: {old_sessions} (count={old_sessions_count})")

    # ========== 2) 将 train_info["acc_matrix"] 载入到 ContinualMetrics 里 ==========
    cm = ContinualMetrics()
    cm.acc_matrix = train_info["acc_matrix"]  # 直接复用

    # ========== 3) 初始化本次训练的 session_info，用来记录训练细节 ==========
    # (你可以按需设计里面的结构)
    session_info = {
        "session_name": args.session_name,
        "task_name": new_task_name,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": args.epochs,
        "details": {},  # 训练过程中收集的数据
        "final_metrics": None,
        "args": vars(args)
    }

    # ========== 4) 创建模型 + (可选) Continual Learning 策略 ==========
    base_model = BaseMultimodalModel(
        args.text_model_name,
        args.image_model_name,
        multimodal_fusion=args.fusion_strategy,
        num_heads=args.num_heads,
        mode=args.mode
    )
    if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
        logger.info(f"Loading pretrained base_model from {args.pretrained_model_path}")
        base_model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)

    head = get_head(new_task_name, base_model, args)
    full_model = Full_Model(base_model, head, dropout_prob=args.dropout_prob)
    full_model.to(device)

    # 若不是第一次，则可以加载 EWC fisher
    ewc = None
    if old_sessions_count > 0 and args.ewc == 1:
        ewc = MultiTaskEWC(
            model=full_model,
            current_task_name=new_task_name,
            ewc_lambda=args.ewc_lambda,
            ewc_dir="ewc_params"  # 你定义的保存EWC参数的目录
        )
        ewc.load_all_previous_tasks()

    replay_memory = None
    if args.replay == 1:
        replay_memory = ExperienceReplayMemory()
        logger.info("进入重放模式")
        # 构造基于动态阈值的重放条件函数，利用已有历史会话的信息
        dynamic_condition = make_dynamic_replay_condition(train_info.get("sessions", []), threshold_factor=0.9)
        # 注册所有历史会话到经验重放中（基于 session 级别）
        for session_info in train_info.get("sessions", []):
            replay_memory.add_session_memory_buffer(
                session_info=session_info,
                memory_percentage=args.memory_percentage,  # 如 0.05，即保存训练集中 5% 的样本
                replay_ratio=0.25,  # 重放批次占当前 batch size 的比例
                replay_frequency=args.replay_frequency,  # 每隔多少步（或 epoch）检查一次
                replay_condition=dynamic_condition  # 使用动态重放条件
            )
        logger.info("加载了 %d 个历史会话用于重放", len(replay_memory.session_memory_buffers))

    # ========== 5) 训练该任务 ==========
    optimizer = AdamW(full_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
    train_dataset = get_dataset(new_task_name, "train", args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    tokenizer = train_dataset.tokenizer
    for batch in train_loader:
        # 输出前两条样例
        for i in range(min(10, batch["input_ids"].size(0))):
            tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i].tolist())
            decoded_tokens = "|".join(tokens)
            print(f"Sample {i} decoded text: {decoded_tokens}")
            print(f"Sample {i} labels: {batch['labels'][i].tolist()}")
        # 输出各字段张量的尺寸
        print("input_ids shape:", batch["input_ids"].shape)
        print("attention_mask shape:", batch["attention_mask"].shape)
        if "token_type_ids" in batch:
            print("token_type_ids shape:", batch["token_type_ids"].shape)
        break  # 只处理第一个 batch

    # 早停逻辑需要
    patience = args.patience

    epoch_losses = []
    dev_metrics_history = []
    try:
        for epoch in range(args.epochs):
            t0 = time.time()
            full_model.train()
            total_loss = 0.0
            label_counter = Counter()
            is_sequence_task = (args.task_name in ["mate", "mner", "mabsa"])  # 举例

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                if "token_type_ids" in batch:
                    token_type_ids = batch["token_type_ids"].to(device)
                else:
                    token_type_ids = None
                image_tensor = batch["image_tensor"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()

                if is_sequence_task:
                    # return_sequence=True
                    # => fused_feat.shape = (batch_size, seq_len, fusion_dim)
                    fused_feat = full_model.base_model(
                        input_ids, attention_mask, token_type_ids, image_tensor,
                        return_sequence=True
                    )
                    logits = full_model.head(fused_feat)  # => (batch_size, seq_len, num_labels)
                    # 由于 token 分布不均，采用加权交叉熵
                    if args.task_name == "mate":
                        class_weights = torch.tensor([1.0, 15.0, 15.0], device=device)
                    elif args.task_name == "mner":
                        class_weights = torch.tensor([0.1, 164.0, 10.0, 270.0, 27.0, 340.0, 16.0, 360.0, 2.0],
                                                     device=device)
                    elif args.task_name == "mabsa":
                        class_weights = torch.tensor([1.0, 3700.0, 234.0, 480.0, 34.0, 786.0, 69.0], device=device)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, args.num_labels),
                        labels.view(-1),
                        weight=class_weights,
                        ignore_index=-100
                    )
                else:
                    # 句级分类: return_sequence=False => (batch_size, fusion_dim)
                    fused_feat = full_model.base_model(
                        input_ids, attention_mask, token_type_ids, image_tensor,
                        return_sequence=False
                    )
                    logits = full_model.head(fused_feat)  # => (batch_size, num_labels)

                    loss = nn.functional.cross_entropy(logits, labels)  # => (batch_size)
                    label_counter.update(labels.cpu().numpy())

                if ewc:
                    loss += ewc.penalty(full_model)

                loss.backward()
                # 可以在此处做梯度裁剪(clip)防梯度爆炸
                torch.nn.utils.clip_grad_norm_(full_model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            epoch_losses.append(avg_loss)

            # 如果开启了经验重放，则进行重放
            if args.replay == 1 and replay_memory.do_replay(epoch + 1):
                replay_session = replay_memory.sample_replay_session(epoch + 1)
                if replay_session is not None:
                    replay_loss = replay_memory.run_replay_step(replay_session, full_model, epoch + 1)
                    logger.info("Replay loss: %.4f", replay_loss.item())

            scheduler.step()

            # 验证集
            dev_metrics = evaluate_single_task(full_model, new_task_name, "dev", device, args)
            dev_metrics_history.append(dev_metrics)

            # # early stopping
            # flag_save = False
            # if dev_metrics["accuracy"] > best_dev_acc:
            #     best_dev_acc = dev_metrics["accuracy"]
            #     no_improve_count = 0
            #     flag_save = True
            #     torch.save(full_model.state_dict(), "checkpoints/best_model.pt")
            # else:
            #     no_improve_count += 1
            #     if no_improve_count >= patience:
            #         logger.info(f"[EarlyStopping] Dev accuracy no improve for {patience} epochs.")
            #         break

            elapsed = (time.time() - t0) / 60
            if is_sequence_task:
                logger.info(f"[Task={new_task_name}] Epoch {epoch + 1}/{args.epochs}, "
                            f"Loss={avg_loss:.4f}, "
                            f"Acc(micro_f1)={dev_metrics['accuracy']:.2f}%, "
                            f"chunk_precision={dev_metrics['chunk_precision']:.2f}%, "
                            f"chunk_recall={dev_metrics['chunk_recall']:.2f}%, "
                            f"chunk_f1={dev_metrics['chunk_f1']:.2f}%, "
                            f"Epoch processed in {elapsed:.4f} minutes.")
            else:
                logger.info(f"[Task={new_task_name}] Epoch {epoch + 1}/{args.epochs}, "
                            f"Loss={avg_loss:.4f}, "
                            f"Acc(micro_f1)={dev_metrics['accuracy']:.2f}%, "
                            f"Pre_macro={dev_metrics['precision_macro']:.2f}%, "
                            f"Recall_macro={dev_metrics['recall_macro']:.2f}%, "
                            f"f1_macro={dev_metrics['f1_macro']:.2f}%, "
                            f"LabelDist={label_counter}%, "
                            f"Epoch processed in {elapsed:.4f} minutes.")

        # ========== 6) 用最佳模型做最终 dev/test 测试 ==========
        # if os.path.exists("checkpoints/best_model.pt") and flag_save:
        if os.path.exists("checkpoints/best_model.pt"):
            full_model.load_state_dict(torch.load("checkpoints/best_model.pt"))
        final_dev_metrics = evaluate_single_task(full_model, new_task_name, "dev", device, args)
        final_test_metrics = evaluate_single_task(full_model, new_task_name, "test", device, args)

        session_info["details"] = {
            "epoch_losses": epoch_losses,
            "dev_metrics_history": dev_metrics_history,
            "final_dev_metrics": final_dev_metrics,
            "final_test_metrics": final_test_metrics
        }

        # ========== 7) 更新 EWC fisher ==========
        if ewc:
            ewc.estimate_and_save_fisher(train_loader, device=device, sample_size=200)

        # ========== 8) 保存最终模型 (可选) ==========
        torch.save(full_model.state_dict(), args.output_model_path)
        logger.info(f"Final model saved => {args.output_model_path}")

        # ========== 9) 将本任务追加到旧任务列表中，并计算 a_{k,j} ==========
        #    只有当不是首次训练(即 old_sessions_count >= 1)，才计算多任务指标
        new_task_index = old_sessions_count  # 0-based
        train_info["tasks"].append(new_task_name)

        # 评估之前所有任务 + 本任务
        all_sessions = train_info["sessions"]  # 现在长度 = old_sessions_count + 1
        performance_list = evaluate_all_learned_tasks(full_model, all_sessions, device, train_info)
        # => [acc_task1, acc_task2, ..., acc_task(new)]
        # 在 cm.acc_matrix 中的行索引就是 new_task_index
        cm.update_acc_matrix(new_task_index, performance_list)

        # 若是第一个任务, 不算持续学习指标
        if len(all_sessions) <= 1:
            logger.info("[Info] This is the first task, skip any CL metrics.")
            final_metrics = {}
        else:
            # 现在一共学了 len(all_sessions) 个任务
            k = len(all_sessions)
            final_metrics = compute_metrics_example(cm, k)
            logger.info(f"Continual Metrics after learning {k} tasks: {final_metrics}")

        session_info["final_metrics"] = final_metrics

        # ========== 10) 覆盖 train_info["acc_matrix"] 并将 session_info 追加到 train_info["sessions"] ==========
        train_info["acc_matrix"] = cm.acc_matrix
        train_info["sessions"].append(session_info)

        # ========== 11) 保存新的 train_info 到 JSON ==========
        with open(args.train_info_json, "w", encoding="utf-8") as f:
            json.dump(train_info, f, indent=2)
        logger.info(f"Updated train_info JSON => {args.train_info_json}")
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}", exc_info=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="masc", help="Name of the new task to train.")
    parser.add_argument("--session_name", type=str, default="default_session",
                        help="Name or ID for this training session")
    parser.add_argument("--train_info_json", type=str, default="checkpoints/train_info.json",
                        help="Path to record train info (tasks, data, metrics, etc.)")
    parser.add_argument("--pretrained_model_path", type=str, default="",
                        help="Path to a pretrained model to continue training")
    parser.add_argument("--output_model_path", type=str, default="checkpoints/model_1.pt")

    parser.add_argument("--train_text_file", type=str, default="data/MASC/twitter2015/train.txt")
    parser.add_argument("--test_text_file", type=str, default="data/MASC/twitter2015/test.txt")
    parser.add_argument("--dev_text_file", type=str, default="data/MASC/twittaer2015/dev.txt")
    parser.add_argument("--image_dir", type=str, default="data/MASC/twitter2015/images")
    parser.add_argument("--text_model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--image_model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--fusion_strategy", type=str, default="multi_head_attention",
                        choices=["concat", "multi_head_attention", "add"])
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_labels", type=int, default=3)  # -1, 0, 1
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--mode", type=str, default="multimodal")  # text_only / multimodal
    parser.add_argument("--ewc_lambda", type=float, default=1000)

    # == 新增正则化和防过拟合的超参 ==
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularization).")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout probability in Full_Model.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping (epochs).")

    # == Continual Learning 相关 ==
    parser.add_argument("--ewc", type=int, default=0, help="whether to use ewc")
    parser.add_argument("--parallel", type=int, default=0, help="whether to use ewc")
    parser.add_argument("--replay", type=int, default=0, help="whether to use experience replay")
    parser.add_argument("--memory_percentage", type=int, default=0.05, help="whether to use experience replay")
    parser.add_argument("--replay_frequency", type=int, default=100, help="whether to use experience replay")
    parser.add_argument("--memory_sampling_strategy", type=str, default='random', choices=['random', 'random-balanced'],
                        help="Strategy for sampling memory buffer samples.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = setup_logger(logging.INFO)  # 或者设置其他等级 DEBUG/ERROR 等
    logger.info("Starting train.py for a single new task ...")
    train(args, logger)


if __name__ == "__main__":
    main()
