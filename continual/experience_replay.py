import os
import argparse
import random
import logging
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable

import torch
from torch.optim import AdamW
from torch.utils.data._utils.collate import default_collate

# 新增导入：用于实时评估当前模型
from modules.evaluate import evaluate_single_task
from datasets.get_dataset import get_dataset  # 使用你现有的 get_dataset 函数
from utils.logging import setup_logger
from models.task_heads.get_head import get_head

logger = logging.getLogger(__name__)

def default_replay_condition(session_info: Dict) -> bool:
    """
    默认的重放条件函数。此处简单返回 True，
    你可以根据 session_info 中的信息自定义逻辑。
    """
    return True

def make_dynamic_replay_condition(all_sessions: list, threshold_factor: float = 0.95) -> Callable[[Dict, torch.nn.Module, torch.device, dict], bool]:
    """
    根据所有历史会话的验证指标，生成一个动态重放条件函数。
    对于每个历史会话，动态条件会通过实时评估当前模型在该任务上的准确率，
    如果当前准确率低于历史记录的最终测试准确率乘以阈值，则认为该会话需要进行重放。

    参数：
        all_sessions: 历史训练会话信息列表，每个 session_info 的 details 中应包含
                      "final_test_metrics" 字典，且其中有 "acc" 字段。
        threshold_factor: 阈值因子，默认为 0.95。

    返回：
        一个函数，该函数接收 (session_info, model, device, args) 并返回布尔值。
    """
    valid_accuracies = []
    for session in all_sessions:
        ftm = session.get("details", {}).get("final_test_metrics")
        if ftm and "acc" in ftm:
            valid_accuracies.append(ftm["acc"])
    best_accuracy = max(valid_accuracies) if valid_accuracies else 1.0

    def dynamic_condition(session_info: Dict, model: torch.nn.Module, device: torch.device, args: dict) -> bool:
        # 通过当前模型重新评估该历史任务的测试集准确率
        # 注意：这里调用 evaluate_single_task，要求 session_info["args"] 包含必要的评估信息
        session_args = session_info.get("args")
        # 将参数转换为 Namespace，如果需要
        if isinstance(session_args, dict):
            session_args = argparse.Namespace(**session_args)
        current_metrics = evaluate_single_task(model, session_info["task_name"], "test", device, session_args)
        current_acc = current_metrics.get("acc", 0.0)
        historical_acc = session_info.get("details", {}).get("final_test_metrics", {}).get("acc", 1.0)
        logger.info("当前评估会话 '%s' 的 acc=%.2f，历史 acc=%.2f",
                    session_info["session_name"], current_acc, historical_acc)
        return current_acc < historical_acc * threshold_factor

    return dynamic_condition

class ExperienceReplayMemory:
    def __init__(self, fisher_dict: Dict[str, torch.Tensor] = None):
        """
        初始化经验重放内存。内部用 session_memory_buffers 记录每个历史训练会话的重放缓冲区。
        """
        self.session_memory_buffers = {}
        self.fisher = fisher_dict
        logger.info("ExperienceReplayMemory 初始化完成 (fisher %sabled)。",
                    "en" if fisher_dict is not None else "dis")

    def add_fisher_info_to_buffer(self, session_info: Dict, memory_size: int):
        """
        如果启用了 Fisher 计算，计算并将 Fisher 敏感度分数添加到缓冲区。
        """
        if self.fisher is not None:
            model = session_info["model"]  # 需在注册时将 model 引入 session_info
            device = session_info["device"]
            model.eval()
            sensitivity = {}

            # 获取完整的数据集和相关参数
            dataset = session_info["dataset"]
            batch_collate_fn = session_info["batch_collate_fn"]

            # 计算每个样本的敏感度
            for idx in range(len(dataset)):
                sample = dataset[idx]
                batch = batch_collate_fn([sample])

                # 移动到设备
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(device)

                logits = model.base_model(
                    batch["input_ids"], batch["attention_mask"],
                    batch.get("token_type_ids", None), batch["image_tensor"],
                    return_sequence=(session_info["task_name"] in ["mate","mner","mabsa"])
                )

                loss = F.cross_entropy(
                    model.set_active_head(logits).view(-1, session_info["num_labels"]),
                    batch["labels"].view(-1),
                    ignore_index=-100
                )

                model.zero_grad()
                loss.backward()

                # 计算敏感度
                score = 0.0
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    Fp = self.fisher.get(name)
                    if Fp is None:
                        continue
                    g2 = (param.grad.detach() ** 2).to(Fp.device)
                    score += (Fp * g2).sum().item()
                sensitivity[idx] = score

            # 根据敏感度对所有样本进行排序，并选择前 `memory_size` 个样本
            sorted_idxs = sorted(sensitivity.keys(), key=lambda i: sensitivity[i], reverse=True)

            # 选择排名靠前的 `memory_size` 个样本作为新的 memory_indices
            selected_indices = sorted_idxs[:memory_size]

            session_info["memory_indices"] = selected_indices

            session_info["sensitivity_scores"] = sensitivity
            logger.info("为会话 '%s' 计算了 %d 个样本的 Fisher 敏感度得分。",
                        session_info["session_name"], len(sensitivity))
        else:
            logger.warning("Fisher 信息未启用，因此不会计算敏感度。")

    def add_session_memory_buffer(self,
                                  session_info: Dict,
                                  memory_percentage: float,
                                  replay_ratio: float = 0.5,
                                  replay_frequency: int = 4,
                                  replay_condition: Callable[[Dict, torch.nn.Module, torch.device, dict], bool] = default_replay_condition):
        """
        注册一个历史训练会话用于经验重放。
        """
        if isinstance(session_info.get("args"), dict):
            session_args = argparse.Namespace(**session_info["args"])
        else:
            session_args = session_info["args"]

        session_name = session_info["session_name"]
        dataset = get_dataset(session_info["task_name"], "train", session_args)
        batch_collate_fn = default_collate
        batch_size = session_args.batch_size

        memory_size = int(memory_percentage * len(dataset))
        if memory_size < 1 and memory_percentage > 0:
            memory_size = 1
        memory_indices = random.sample(range(len(dataset)), memory_size)

        self.session_memory_buffers[session_name] = {
            "session_info": session_info,
            "dataset": dataset,
            "batch_collate_fn": batch_collate_fn,
            "batch_size": batch_size,
            "memory_indices": memory_indices,
            "replay_ratio": replay_ratio,
            "replay_frequency": replay_frequency,
            "replay_condition": replay_condition
        }
        logger.info("为会话 '%s' 创建了重放内存缓冲区，共包含 %d 个样本。",
                    session_name, len(memory_indices))
        self.add_fisher_info_to_buffer(session_info, memory_size)


    def do_replay(self, current_step: int, model: torch.nn.Module, device: torch.device, args: dict) -> bool:
        """
        判断当前步数是否有任何历史会话满足重放条件。
        """
        eligible_sessions = []
        for session_name, buffer in self.session_memory_buffers.items():
            replay_frequency = buffer["replay_frequency"]
            replay_condition = buffer["replay_condition"]
            # 这里将当前模型、设备、参数传递进去
            if current_step % replay_frequency == 0 and replay_condition(buffer["session_info"], model, device, args):
                eligible_sessions.append(session_name)
        if eligible_sessions:
            logger.info("在步数 %d，有会话 %s 满足重放条件。", current_step, eligible_sessions)
            return True
        else:
            logger.info("在步数 %d，没有会话满足重放条件。", current_step)
            return False

    def sample_replay_session(self, current_step: int, model: torch.nn.Module, device: torch.device, args: dict) -> str:
        """
        从满足重放条件的会话中随机采样一个进行重放。
        """
        eligible_sessions = []
        for session_name, buffer in self.session_memory_buffers.items():
            replay_frequency = buffer["replay_frequency"]
            replay_condition = buffer["replay_condition"]
            if current_step % replay_frequency == 0 and replay_condition(buffer["session_info"], model, device, args):
                eligible_sessions.append(session_name)
        if not eligible_sessions:
            logger.warning("在步数 %d，没有满足重放条件的会话。", current_step)
            return None
        sampled_session = random.choice(eligible_sessions)
        logger.info("在步数 %d，采样到会话 '%s' 进行重放。", current_step, sampled_session)
        return sampled_session

    def sample_replay_batch(self, session_name: str) -> Dict:
        """
        从指定会话的重放缓冲区采样一个批次数据。
        """
        if session_name not in self.session_memory_buffers:
            logger.error("会话 '%s' 未注册，无法采样重放批次。", session_name)
            return None
        buffer = self.session_memory_buffers[session_name]
        dataset = buffer["dataset"]
        batch_collate_fn = buffer["batch_collate_fn"]
        total_samples = len(dataset)
        current_batch_size = buffer["batch_size"]
        replay_batch_size = int(buffer["replay_ratio"] * current_batch_size)
        if replay_batch_size < 1 and buffer["replay_ratio"] > 0:
            replay_batch_size = 1
        # —— 若有敏感度得分，则 top-k —— #
        scores = buffer.get("sensitivity_scores")
        if scores:
            # 按得分排序 memory_indices
            sorted_idxs = sorted(buffer["memory_indices"],
                                 key=lambda i: scores.get(i, 0.0),
                                 reverse=True)
            sampled_indices = sorted_idxs[:replay_batch_size]
        else:
            # 回退到随机采样
            sampled_indices = random.sample(range(total_samples), replay_batch_size)
        logger.info("会话 '%s' 重放批次采样索引: %s", session_name, sampled_indices)
        batch = batch_collate_fn([dataset[i] for i in sampled_indices])
        return batch

    def create_optimizer(self, model, session_name: str):
        """
        为指定会话的重放创建优化器。
        """
        logger.info("为会话 '%s' 创建重放优化器。", session_name)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        lr = 5e-4
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))
        logger.info("会话 '%s' 的重放优化器创建成功，学习率=%.4f", session_name, lr)
        return optimizer

    def run_replay_step(self, session_name: str, model, current_step: int, device: torch.device,
                        args: dict) -> torch.Tensor:
        """
        针对指定会话，在当前训练步数下执行一次重放步骤。
        这里提取 replay_batch 中的必要输入，并计算交叉熵损失。
        """
        if session_name not in self.session_memory_buffers:
            logger.error("会话 '%s' 未注册到重放内存中。", session_name)
            return None
        logger.info("会话 '%s' 在步数 %d 开始执行重放步骤。", session_name, current_step)

        # 创建该会话的专用重放优化器
        optimizer = self.create_optimizer(model, session_name)

        # 从重放缓冲区中采样一个批次数据
        replay_batch = self.sample_replay_batch(session_name)

        # 将 replay_batch 中的各个张量移动到 device 上
        for key, value in replay_batch.items():
            if torch.is_tensor(value):
                replay_batch[key] = value.to(device)

        # 提取模型 forward 所需的输入
        input_ids = replay_batch["input_ids"]
        attention_mask = replay_batch["attention_mask"]
        token_type_ids = replay_batch.get("token_type_ids", None)
        image_tensor = replay_batch["image_tensor"]
        labels = replay_batch["labels"]


        # 获取历史任务的参数信息
        session_info = self.session_memory_buffers[session_name]["session_info"]
        if isinstance(session_info.get("args"), dict):
            session_args = argparse.Namespace(**session_info["args"])
        else:
            session_args = session_info["args"]
        
        task_name = session_args.task_name
        num_labels = session_args.num_labels


        # 判断是否为序列任务（例如 MNER、MATE、MABSA）
        if task_name in ["mate", "mner", "mabsa"]:
            fused_feat = model.base_model(
                input_ids, attention_mask, token_type_ids, image_tensor,
                return_sequence=True
            )

            # 使用保存的历史任务head
            if hasattr(model, 'set_active_head'):
                model.set_active_head(session_name)
                logits = model.head(fused_feat)  # => (batch_size, seq_len, num_labels)
            else:
                # 如果没有set_active_head方法，回退到创建新head
                tmp_head = get_head(task_name, model.base_model, session_args).to(device)
                logits = tmp_head(fused_feat)  # => (batch_size, seq_len, num_labels)



            # 扁平化 logits 和 labels
            logits = logits.view(-1, num_labels)  # 扁平化 logits 为 (batch_size * seq_len, num_labels)
            labels = labels.view(-1)  # 扁平化 labels 为 (batch_size * seq_len)


            # 根据任务选择对应的 class_weights（如果需要使用权重的话）
            if task_name == "mate":
                class_weights = torch.tensor([1.0, 15.0, 15.0], device=device)
            elif task_name == "mner":
                class_weights = torch.tensor([0.1, 164.0, 10.0, 270.0, 27.0, 340.0, 16.0, 360.0, 2.0],
                                                 device=device)
            elif task_name == "mabsa":
                class_weights = torch.tensor([1.0, 3700.0, 234.0, 480.0, 34.0, 786.0, 69.0], device=device)
            loss = nn.functional.cross_entropy(
                logits,
                labels,
                weight=class_weights,
                ignore_index=-100
            )
        else:
            # 句级分类: return_sequence=False => (batch_size, fusion_dim)
            fused_feat = model.base_model(
                input_ids, attention_mask, token_type_ids, image_tensor,
                return_sequence=False
            )
            
            # 使用保存的历史任务head
            if hasattr(model, 'set_active_head'):
                model.set_active_head(session_name)
                logits = model.head(fused_feat)  # => (batch_size, num_labels)
            else:
                # 如果没有set_active_head方法，回退到创建新head
                tmp_head = get_head(task_name, model.base_model, session_args).to(device)
                logits = tmp_head(fused_feat)  # => (batch_size, num_labels)

            loss = nn.functional.cross_entropy(logits, labels)  # => (batch_size)

        loss.backward()
        optimizer.step()

        # 恢复当前任务的head（如果模型支持）
        if hasattr(model, 'set_active_head') and hasattr(args, 'session_name'):
            try:
                model.set_active_head(args.session_name)
            except Exception as e:
                logger.warning(f"Failed to restore current task head: {e}")

        logger.info("会话 '%s' 在步数 %d 重放步骤完成，损失值: %.4f", session_name, current_step, loss.item())
        return loss
