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
from utils.ensureFileExists import ensure_directory_exists
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
    ensure_directory_exists(args.train_info_json)
    ensure_directory_exists(args.ewc_dir)
    ensure_directory_exists(args.output_model_path)

    logger.info(f"=== Start training for new task: {new_task_name} ===")
    # ========== 1) 从 JSON 加载或初始化 train_info ==========
    train_info = {}
    if os.path.exists(args.train_info_json):
        with open(args.train_info_json, "r", encoding="utf-8") as f:
            try:
                train_info = json.load(f)
                logger.info(f"Loaded existing training info")
            except:
                logger.warning(f"Failed to load, using empty info.")
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
    # logger.info(f"Previously learned sessions: {old_sessions} (count={old_sessions_count})")

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
        "args": vars(args),
        "fisher_file": os.path.join(args.ewc_dir, f"{args.session_name}_fisher.pt")  # 保存 Fisher 文件路径

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

    # 初始化 EWC
    ewc = None
    if args.ewc == 1:  # 检查是否启用了 EWC
        logger.info("进入 EWC 模式")
        ewc = MultiTaskEWC(
            model=full_model,
            current_task_name=new_task_name,
            session_name=args.session_name,
            num_labels=args.num_labels,
            ewc_lambda=args.ewc_lambda,
            ewc_dir=args.ewc_dir  # 可以直接传递到 ewc_dir
        )
        if old_sessions_count > 0:
            ewc.load_all_previous_tasks(train_info)  # 加载历史任务的 Fisher 和 optpar 数据

    replay_memory = None
    if args.replay == 1:
        replay_memory = ExperienceReplayMemory()
        logger.info("进入重放模式")
        # 构造基于动态阈值的重放条件函数，利用已有历史会话的信息
        dynamic_condition = make_dynamic_replay_condition(train_info.get("sessions", []), threshold_factor=0.9)
        # 注册所有历史会话到经验重放中（基于 session 级别）
        for hist_session in train_info.get("sessions", []):
            replay_memory.add_session_memory_buffer(
                session_info=hist_session,
                memory_percentage=args.memory_percentage,  # 如 0.05
                replay_ratio=0.25,
                replay_frequency=args.replay_frequency,
                replay_condition=dynamic_condition
            )
        logger.info("加载了 %d 个历史会话用于重放", len(replay_memory.session_memory_buffers))

    # ========== 5) 训练该任务 ==========
    optimizer = AdamW(full_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    train_dataset = get_dataset(new_task_name, "train", args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    tokenizer = train_dataset.tokenizer
    # for batch in train_loader:
    #     # 输出前两条样例
    #     for i in range(min(2, batch["input_ids"].size(0))):
    #         tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i].tolist())
    #         decoded_tokens = "|".join(tokens)
    #         print(f"Sample {i} decoded text: {decoded_tokens}")
    #         print(f"Sample {i} labels: {batch['labels'][i].tolist()}")
    #     # 输出各字段张量的尺寸
    #     print("input_ids shape:", batch["input_ids"].shape)
    #     print("attention_mask shape:", batch["attention_mask"].shape)
    #     if "token_type_ids" in batch:
    #         print("token_type_ids shape:", batch["token_type_ids"].shape)
    #     break  # 只处理第一个 batch

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
            if args.replay == 1 and replay_memory.do_replay(epoch + 1, full_model, device, args):
                replay_session_name = replay_memory.sample_replay_session(epoch + 1, full_model, device, args)
                if replay_session_name is not None:
                    # 从历史会话中查找对应的 session_info，提取该任务的 args
                    replay_session_info = next(
                        (s for s in train_info["sessions"] if s["session_name"] == replay_session_name), None)
                    if replay_session_info is not None:
                        replay_args = replay_session_info["args"]
                    else:
                        # 如果找不到，还是采用当前 args（或抛出异常，根据需要处理）
                        replay_args = vars(args)
                    # 记录下来每次触发重放的会话名称
                    if "replay_sessions" not in session_info["details"]:
                        session_info["details"]["replay_sessions"] = []
                    session_info["details"]["replay_sessions"].append(replay_session_name)
                    replay_loss = replay_memory.run_replay_step(replay_session_name, full_model, epoch + 1, device,
                                                                replay_args)
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
            logger.info(f"[Task={new_task_name}] Epoch {epoch + 1}/{args.epochs}, "
                        f"Loss={avg_loss:.4f}, "
                        f"Acc(micro_f1)={dev_metrics['acc']:.2f}%, "
                        f"micro_precision={dev_metrics['micro_prec']:.2f}%, "
                        f"micro_recall={dev_metrics['micro_recall']:.2f}%, "
                        f"micro_f1={dev_metrics['micro_f1']:.2f}%, "
                        f"Epoch processed in {elapsed:.4f} minutes.")
            # if is_sequence_task:
            #     logger.info(f"[Task={new_task_name}] Epoch {epoch + 1}/{args.epochs}, "
            #                 f"Loss={avg_loss:.4f}, "
            #                 f"Acc(micro_f1)={dev_metrics['accuracy']:.2f}%, "
            #                 f"chunk_precision={dev_metrics['chunk_precision']:.2f}%, "
            #                 f"chunk_recall={dev_metrics['chunk_recall']:.2f}%, "
            #                 f"chunk_f1={dev_metrics['chunk_f1']:.2f}%, "
            #                 f"Epoch processed in {elapsed:.4f} minutes.")
            # else:
            #     logger.info(f"[Task={new_task_name}] Epoch {epoch + 1}/{args.epochs}, "
            #                 f"Loss={avg_loss:.4f}, "
            #                 f"Acc(micro_f1)={dev_metrics['accuracy']:.2f}%, "
            #                 f"Pre_macro={dev_metrics['precision_macro']:.2f}%, "
            #                 f"Recall_macro={dev_metrics['recall_macro']:.2f}%, "
            #                 f"f1_macro={dev_metrics['f1_macro']:.2f}%, "
            #                 f"LabelDist={label_counter}%, "
            #                 f"Epoch processed in {elapsed:.4f} minutes.")

        # ========== 6) 用最佳模型做最终 dev/test 测试 ==========
        # if os.path.exists("checkpoints/best_model.pt") and flag_save:
        # if os.path.exists("checkpoints/best_model.pt"):
        #     full_model.load_state_dict(torch.load("checkpoints/best_model.pt"))
        final_dev_metrics = evaluate_single_task(full_model, new_task_name, "dev", device, args)
        final_test_metrics = evaluate_single_task(full_model, new_task_name, "test", device, args)


        # ========== 7) 更新 EWC fisher ==========
        session_info["fisher_file"] = os.path.join(args.ewc_dir, f"{args.session_name}_fisher.pt")

        if ewc:
            ewc.estimate_and_save_fisher(train_loader, device=device, sample_size=200)
            #
            # # 保存 EWC 文件
            # fisher_path = os.path.join(args.ewc_dir, f"{args.session_name}_fisher.pt")
            # torch.save({
            #     'fisher': ewc.fisher_all,
            #     'optpar': ewc.optpar_all
            # }, fisher_path)
            # logger.info(f"Saved Fisher and Optpar for task={new_task_name} => {fisher_path}")

        # ========== 8) 保存最终模型 (可选) ==========
        torch.save(full_model.state_dict(), args.output_model_path)
        logger.info(f"Final model saved => {args.output_model_path}")

        # ========== 9) 将本任务追加到旧任务列表中，并计算 a_{k,j} ==========
        #    只有当不是首次训练(即 old_sessions_count >= 1)，才计算多任务指标
        new_task_index = old_sessions_count  # 0-based
        train_info["tasks"].append(new_task_name)

        # 评估之前所有任务在当前模型下的测试准确率
        previous_performance_list = evaluate_all_learned_tasks(full_model, train_info["sessions"], device, train_info)
        # 将之前任务的准确率与当前任务的准确率拼接成完整的 performance_list
        performance_list = previous_performance_list + [final_test_metrics['acc']]

        # 更新 acc_matrix，第 new_task_index 行对应的所有任务的测试准确率
        cm.update_acc_matrix(new_task_index, performance_list)

        # 若是第一个任务, 不算持续学习指标
        if len(train_info["sessions"]) + 1 <= 1:
            logger.info("[Info] This is the first task, skip any CL metrics.")
            final_metrics = {}
        else:
            k = len(train_info["sessions"]) + 1  # 总任务数
            final_metrics = compute_metrics_example(cm, k)
            logger.info(f"Continual Metrics after learning {k} tasks: {final_metrics}")

        session_info["final_metrics"] = final_metrics
        session_info["details"].update({
            "epoch_losses": epoch_losses,
            "dev_metrics_history": dev_metrics_history,
            "final_dev_metrics": final_dev_metrics,
            "final_test_metrics": final_test_metrics
        })
        # ========== 10) 更新 train_info ==========
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
    parser.add_argument("--dev_text_file", type=str, default="data/MASC/twitter2015/dev.txt")
    parser.add_argument("--image_dir", type=str, default="data/img")
    parser.add_argument("--text_model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--image_model_name", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--fusion_strategy", type=str, default="multi_head_attention",
                        choices=["concat", "multi_head_attention", "add"])
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)  # 1e-5
    parser.add_argument("--epochs", type=int, default=20)  # 5
    parser.add_argument("--num_labels", type=int, default=3)  # -1, 0, 1
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--step_size", type=int, default=10)  # 2
    parser.add_argument("--gamma", type=float, default=0.5)  # 0.1

    parser.add_argument("--mode", type=str, default="multimodal")  # text_only / multimodal

    # == 新增正则化和防过拟合的超参 ==
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularization).")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout probability in Full_Model.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping (epochs).")

    # == Continual Learning 相关 ==
    parser.add_argument("--ewc", type=int, default=0, help="whether to use ewc")
    parser.add_argument("--ewc_dir", type=str, default="ewc_params", help="Directory to save EWC params")
    parser.add_argument("--ewc_lambda", type=float, default=100)
    parser.add_argument("--parallel", type=int, default=0, help="whether to use ewc")
    parser.add_argument("--replay", type=int, default=1, help="whether to use experience replay")
    parser.add_argument("--memory_percentage", type=int, default=0.05, help="whether to use experience replay")
    parser.add_argument("--replay_frequency", type=int, default=5, help="whether to use experience replay")
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
