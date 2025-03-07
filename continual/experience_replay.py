import os
import argparse
import random
import logging
from typing import Dict, Callable

import torch
from torch.optim import AdamW
from torch.utils.data._utils.collate import default_collate

from datasets.get_dataset import get_dataset  # 使用你现有的 get_dataset 函数
from utils.logging import setup_logger

logger = logging.getLogger(__name__)

def default_replay_condition(session_info: Dict) -> bool:
    """
    默认的重放条件函数。此处简单返回 True，
    你可以根据 session_info 中的信息（例如训练效果、持续遗忘情况等）自定义逻辑，
    比如当某个会话的最终指标低于预期时返回 True。
    """
    return True


def make_dynamic_replay_condition(all_sessions: list, threshold_factor: float = 0.95) -> Callable[[Dict], bool]:
    """
    根据所有历史会话的验证指标，生成一个动态重放条件函数。
    对于每个会话，如果该会话的最终验证准确率低于所有会话中最佳准确率的 threshold_factor 倍，则认为该会话需要进行重放。

    参数：
        all_sessions: 包含历史训练会话信息的列表，每个 session_info 至少应包含 "final_metrics" 字典，且其中有 "accuracy" 字段。
        threshold_factor: 阈值因子，默认为 0.95，即要求准确率不低于最佳准确率的 95%。

    返回：
        一个函数，该函数接收一个 session_info（字典）并返回布尔值。
    """
    valid_accuracies = []
    for session in all_sessions:
        fm = session.get("final_metrics")
        if fm and "accuracy" in fm:
            valid_accuracies.append(fm["accuracy"])
    # 若没有有效的验证指标，则默认最佳准确率为 1.0
    best_accuracy = max(valid_accuracies) if valid_accuracies else 1.0

    def dynamic_condition(session_info: Dict) -> bool:
        fm = session_info.get("final_metrics")
        if not fm or "accuracy" not in fm:
            # 若历史记录中没有验证指标，则不触发重放
            return False
        # 当该会话的准确率低于最佳准确率的 threshold_factor 倍时，触发重放
        return fm["accuracy"] < best_accuracy * threshold_factor

    return dynamic_condition

class ExperienceReplayMemory:
    def __init__(self):
        """
        基于会话级别初始化经验重放内存。
        内部用 session_memory_buffers 字典记录每个历史训练会话的重放缓冲区，
        key 为 session_name，value 为该会话对应的缓冲信息。
        """
        self.session_memory_buffers = {}
        logger.info("ExperienceReplayMemory 初始化完成，session replay memory 为空。")

    def add_session_memory_buffer(self,
                                  session_info: Dict,
                                  memory_percentage: float,
                                  replay_ratio: float = 0.25,
                                  replay_frequency: int = 100,
                                  replay_condition: Callable[[Dict], bool] = default_replay_condition):
        """
        注册一个历史训练会话用于经验重放。
        参数：
          - session_info (Dict): 包含至少 'session_name'、'task_name'、'args'（字典形式）的训练会话信息
          - memory_percentage (float): 用于重放的样本比例（例如 0.05 表示存储训练集中 5% 的数据）
          - replay_ratio (float): 重放批次中 replay 数据所占比例，相对于当前 batch_size（默认 0.25）
          - replay_frequency (int): 每隔多少步（或 epoch）执行一次重放
          - replay_condition (callable): 判断是否满足重放条件的函数，接收 session_info 返回 bool
        """
        # 将 session_info["args"] 转换为 Namespace 对象（若为字典）
        if isinstance(session_info.get("args"), dict):
            session_args = argparse.Namespace(**session_info["args"])
        else:
            session_args = session_info["args"]

        session_name = session_info["session_name"]
        # 使用历史会话中的 task_name 加载数据集，假设训练数据与当前一致
        dataset = get_dataset(session_info["task_name"], "train", session_args)
        batch_collate_fn = default_collate
        batch_size = session_args.batch_size

        memory_size = int(memory_percentage * len(dataset))
        if memory_size < 1 and memory_percentage > 0:
            memory_size = 1
        # 随机采样用于重放的数据索引
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

    def do_replay(self, current_step: int) -> bool:
        """
        判断当前步数是否有任何历史会话满足重放条件。
        遍历所有注册的会话，如果当前步数满足该会话设置的 replay_frequency 且 replay_condition 返回 True，
        则认为满足重放条件。
        """
        eligible_sessions = []
        for session_name, buffer in self.session_memory_buffers.items():
            replay_frequency = buffer["replay_frequency"]
            replay_condition = buffer["replay_condition"]
            if current_step % replay_frequency == 0 and replay_condition(buffer["session_info"]):
                eligible_sessions.append(session_name)
        if eligible_sessions:
            logger.info("在步数 %d，有会话 %s 满足重放条件。", current_step, eligible_sessions)
            return True
        else:
            logger.info("在步数 %d，没有会话满足重放条件。", current_step)
            return False

    def sample_replay_session(self, current_step: int) -> str:
        """
        从满足重放条件的会话中随机采样一个进行重放。
        如果没有满足条件的会话，返回 None。
        """
        eligible_sessions = []
        for session_name, buffer in self.session_memory_buffers.items():
            replay_frequency = buffer["replay_frequency"]
            replay_condition = buffer["replay_condition"]
            if current_step % replay_frequency == 0 and replay_condition(buffer["session_info"]):
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
        重放批次大小 = replay_ratio * 当前 batch_size（至少采样 1 个样本）。
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
        sampled_indices = random.sample(range(total_samples), replay_batch_size)
        logger.info("会话 '%s' 重放批次采样索引: %s", session_name, sampled_indices)
        batch = batch_collate_fn([dataset[i] for i in sampled_indices])
        return batch

    def create_optimizer(self, model, session_name: str):
        """
        为指定会话的重放创建优化器（可与正常训练优化器不同）。
        """
        logger.info("为会话 '%s' 创建重放优化器。", session_name)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        lr = 1e-4
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))
        logger.info("会话 '%s' 的重放优化器创建成功，学习率=%.4f", session_name, lr)
        return optimizer

    def run_replay_step(self, session_name: str, model, current_step: int) -> torch.Tensor:
        """
        针对指定会话，在当前训练步数下执行一次重放步骤（如果满足条件）。
        参数：
          - session_name: 历史训练会话的唯一标识符
          - model: 模型，要求 forward 方法接收 batch 并返回损失
          - current_step: 当前训练步数（或 epoch）
        返回：
          - 重放步骤计算的损失
        """
        if session_name not in self.session_memory_buffers:
            logger.error("会话 '%s' 未注册到重放内存中。", session_name)
            return None
        logger.info("会话 '%s' 在步数 %d 开始执行重放步骤。", session_name, current_step)
        optimizer = self.create_optimizer(model, session_name)
        replay_batch = self.sample_replay_batch(session_name)
        loss = model(replay_batch)  # 这里假定模型 forward 返回损失
        loss.backward()
        optimizer.step()
        logger.info("会话 '%s' 在步数 %d 重放步骤完成，损失值: %.4f", session_name, current_step, loss.item())
        return loss
