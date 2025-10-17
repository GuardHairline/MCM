# continual/experience_replay_fixed.py
"""
修复后的Experience Replay实现

主要修复：
1. 移除硬编码的类别权重，使用label_config
2. 使用label_config判断任务类型
3. 修复重放步骤中的优化器使用
4. 改进采样策略
5. 支持label_embedding
6. 修复head切换逻辑
"""

import os
import argparse
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable, Optional
from torch.utils.data._utils.collate import default_collate

# 新增导入：用于实时评估当前模型
from modules.evaluate import evaluate_single_task
from datasets.get_dataset import get_dataset
from continual.label_config import get_label_manager

logger = logging.getLogger(__name__)


def default_replay_condition(session_info: Dict, model=None, device=None, args=None) -> bool:
    """默认的重放条件函数，简单返回 True"""
    return True


def make_dynamic_replay_condition(all_sessions: list, threshold_factor: float = 0.95) -> Callable:
    """
    根据历史会话的验证指标，生成动态重放条件函数
    
    改进：
    - 添加异常处理
    - 更清晰的日志
    """
    valid_accuracies = []
    for session in all_sessions:
        ftm = session.get("details", {}).get("final_test_metrics")
        if ftm and "acc" in ftm:
            valid_accuracies.append(ftm["acc"])
    
    best_accuracy = max(valid_accuracies) if valid_accuracies else 1.0
    
    def dynamic_condition(session_info: Dict, model: torch.nn.Module, 
                         device: torch.device, args: dict) -> bool:
        try:
            # 通过当前模型重新评估该历史任务的测试集准确率
            session_args = session_info.get("args")
            if isinstance(session_args, dict):
                session_args = argparse.Namespace(**session_args)
            
            current_metrics = evaluate_single_task(
                model, session_info["task_name"], "dev", device, session_args
            )
            current_acc = current_metrics.get("acc", 0.0)
            historical_acc = session_info.get("details", {}).get("final_test_metrics", {}).get("acc", 1.0)
            
            should_replay = current_acc < historical_acc * threshold_factor
            
            if should_replay:
                logger.info(
                    f"Session '{session_info['session_name']}': "
                    f"current_acc={current_acc:.4f} < {historical_acc * threshold_factor:.4f} "
                    f"(historical_acc={historical_acc:.4f} * {threshold_factor}), needs replay"
                )
            
            return should_replay
        
        except Exception as e:
            logger.warning(f"Dynamic condition evaluation failed for {session_info.get('session_name')}: {e}")
            return True  # 出错时默认重放
    
    return dynamic_condition


class ExperienceReplayMemory:
    """
    经验重放内存管理器（修复版）
    
    改进：
    - 使用label_config获取类别权重
    - 改进采样策略
    - 修复重放训练逻辑
    - 更好的head管理
    """
    
    def __init__(self, fisher_dict: Optional[Dict[str, torch.Tensor]] = None):
        """初始化经验重放内存"""
        self.session_memory_buffers = {}
        self.fisher = fisher_dict
        logger.info(f"ExperienceReplayMemory initialized (Fisher {'enabled' if fisher_dict else 'disabled'})")
    
    def _is_sequence_task(self, task_name: str) -> bool:
        """判断是否为序列任务"""
        return get_label_manager().is_token_level_task(task_name)
    
    def _get_class_weights(self, task_name: str, device: torch.device) -> Optional[torch.Tensor]:
        """获取类别权重"""
        weights = get_label_manager().get_class_weights(task_name)
        if weights is not None:
            return torch.tensor(weights, dtype=torch.float32, device=device)
        return None
    
    def add_fisher_info_to_buffer(self, session_info: Dict, memory_size: int):
        """
        计算Fisher敏感度分数（如果启用）
        
        改进：
        - 使用label_config判断任务类型
        - 更好的异常处理
        - 批量处理提高效率
        """
        if self.fisher is None:
            logger.info("Fisher not enabled, skipping sensitivity calculation")
            return
        
        try:
            model = session_info["model"]
            device = session_info["device"]
            task_name = session_info["task_name"]
            dataset = session_info["dataset"]
            batch_collate_fn = session_info["batch_collate_fn"]
            
            model.eval()
            sensitivity = {}
            
            is_seq_task = self._is_sequence_task(task_name)
            
            logger.info(f"Calculating Fisher sensitivity for session '{session_info['session_name']}' ({len(dataset)} samples)")
            
            # 批量计算以提高效率
            batch_size = 8
            for start_idx in range(0, len(dataset), batch_size):
                end_idx = min(start_idx + batch_size, len(dataset))
                batch_samples = [dataset[i] for i in range(start_idx, end_idx)]
                batch = batch_collate_fn(batch_samples)
                
                # 移动到设备
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(device)
                
                # 前向传播
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                token_type_ids = batch.get("token_type_ids")
                image_tensor = batch["image_tensor"]
                labels = batch["labels"]
                
                fused_feat = model.base_model(
                    input_ids, attention_mask, token_type_ids, image_tensor,
                    return_sequence=is_seq_task
                )
                
                # 设置正确的head
                if hasattr(model, 'set_active_head'):
                    model.set_active_head(session_info['session_name'], strict=False)
                
                logits = model.head(fused_feat)
                
                # 计算损失
                if is_seq_task:
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100
                    )
                else:
                    loss = F.cross_entropy(logits, labels)
                
                model.zero_grad()
                loss.backward()
                
                # 计算每个样本的敏感度（简化为批次敏感度）
                batch_score = 0.0
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    Fp = self.fisher.get(name)
                    if Fp is None:
                        continue
                    g2 = (param.grad.detach() ** 2).to(Fp.device)
                    batch_score += (Fp * g2).sum().item()
                
                # 分配给批次中的每个样本
                for i in range(start_idx, end_idx):
                    sensitivity[i] = batch_score / (end_idx - start_idx)
            
            # 根据敏感度排序并选择top-k
            sorted_idxs = sorted(sensitivity.keys(), key=lambda i: sensitivity[i], reverse=True)
            selected_indices = sorted_idxs[:memory_size]
            
            session_info["memory_indices"] = selected_indices
            session_info["sensitivity_scores"] = sensitivity
            
            logger.info(
                f"Calculated Fisher sensitivity for {len(sensitivity)} samples, "
                f"selected top-{len(selected_indices)}"
            )
        
        except Exception as e:
            logger.error(f"Error calculating Fisher sensitivity: {e}")
            import traceback
            traceback.print_exc()
    
    def add_session_memory_buffer(self,
                                  session_info: Dict,
                                  memory_percentage: float,
                                  replay_ratio: float = 0.5,
                                  replay_frequency: int = 4,
                                  replay_condition: Callable = default_replay_condition):
        """
        注册历史训练会话用于经验重放
        
        改进：
        - 更好的参数验证
        - 改进的采样策略
        """
        try:
            if isinstance(session_info.get("args"), dict):
                session_args = argparse.Namespace(**session_info["args"])
            else:
                session_args = session_info["args"]
            
            session_name = session_info["session_name"]
            dataset = get_dataset(session_info["task_name"], "train", session_args)
            batch_collate_fn = default_collate
            batch_size = session_args.batch_size
            
            memory_size = max(1, int(memory_percentage * len(dataset)))
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
            
            logger.info(
                f"Created replay buffer for session '{session_name}': "
                f"{len(memory_indices)} samples ({memory_percentage*100:.1f}% of {len(dataset)})"
            )
            
            # 如果启用Fisher，计算敏感度
            if self.fisher is not None:
                self.add_fisher_info_to_buffer(session_info, memory_size)
        
        except Exception as e:
            logger.error(f"Error adding session memory buffer: {e}")
            import traceback
            traceback.print_exc()
    
    def do_replay(self, current_step: int, model: torch.nn.Module, 
                  device: torch.device, args: dict) -> bool:
        """判断当前步数是否需要重放"""
        eligible_sessions = []
        for session_name, buffer in self.session_memory_buffers.items():
            replay_frequency = buffer["replay_frequency"]
            replay_condition = buffer["replay_condition"]
            
            if current_step % replay_frequency == 0:
                if replay_condition(buffer["session_info"], model, device, args):
                    eligible_sessions.append(session_name)
        
        if eligible_sessions:
            logger.debug(f"Step {current_step}: eligible sessions for replay: {eligible_sessions}")
            return True
        else:
            return False
    
    def sample_replay_session(self, current_step: int, model: torch.nn.Module, 
                             device: torch.device, args: dict) -> Optional[str]:
        """从符合条件的会话中采样一个进行重放"""
        eligible_sessions = []
        for session_name, buffer in self.session_memory_buffers.items():
            replay_frequency = buffer["replay_frequency"]
            replay_condition = buffer["replay_condition"]
            
            if current_step % replay_frequency == 0:
                if replay_condition(buffer["session_info"], model, device, args):
                    eligible_sessions.append(session_name)
        
        if not eligible_sessions:
            logger.warning(f"Step {current_step}: no eligible sessions for replay")
            return None
        
        sampled_session = random.choice(eligible_sessions)
        logger.debug(f"Step {current_step}: sampled session '{sampled_session}' for replay")
        return sampled_session
    
    def sample_replay_batch(self, session_name: str) -> Optional[Dict]:
        """
        从指定会话的重放缓冲区采样批次数据
        
        改进：
        - 优先使用Fisher敏感度采样
        - 回退到随机采样
        """
        if session_name not in self.session_memory_buffers:
            logger.error(f"Session '{session_name}' not registered")
            return None
        
        buffer = self.session_memory_buffers[session_name]
        dataset = buffer["dataset"]
        batch_collate_fn = buffer["batch_collate_fn"]
        current_batch_size = buffer["batch_size"]
        replay_batch_size = max(1, int(buffer["replay_ratio"] * current_batch_size))
        
        # 优先使用Fisher敏感度采样
        scores = buffer.get("sensitivity_scores")
        if scores:
            # 按敏感度排序
            sorted_idxs = sorted(
                buffer["memory_indices"],
                key=lambda i: scores.get(i, 0.0),
                reverse=True
            )
            sampled_indices = sorted_idxs[:replay_batch_size]
        else:
            # 随机采样
            memory_indices = buffer["memory_indices"]
            sampled_indices = random.sample(
                memory_indices, 
                min(replay_batch_size, len(memory_indices))
            )
        
        logger.debug(f"Session '{session_name}' replay batch: {len(sampled_indices)} samples")
        batch = batch_collate_fn([dataset[i] for i in sampled_indices])
        return batch
    
    def compute_replay_loss(self, session_name: str, model: torch.nn.Module, 
                           device: torch.device) -> Optional[torch.Tensor]:
        """
        计算重放损失（不更新模型）
        
        改进：
        - 使用label_config获取类别权重
        - 正确的head切换
        - 不创建独立优化器
        """
        if session_name not in self.session_memory_buffers:
            logger.error(f"Session '{session_name}' not registered")
            return None
        
        try:
            # 采样重放批次
            replay_batch = self.sample_replay_batch(session_name)
            if replay_batch is None:
                return None
            
            # 移动到设备
            for key, value in replay_batch.items():
                if torch.is_tensor(value):
                    replay_batch[key] = value.to(device)
            
            # 提取输入
            input_ids = replay_batch["input_ids"]
            attention_mask = replay_batch["attention_mask"]
            token_type_ids = replay_batch.get("token_type_ids")
            image_tensor = replay_batch["image_tensor"]
            labels = replay_batch["labels"]
            
            # 获取任务信息
            session_info = self.session_memory_buffers[session_name]["session_info"]
            if isinstance(session_info.get("args"), dict):
                session_args = argparse.Namespace(**session_info["args"])
            else:
                session_args = session_info["args"]
            
            task_name = session_args.task_name
            is_seq_task = self._is_sequence_task(task_name)
            
            # 前向传播
            fused_feat = model.base_model(
                input_ids, attention_mask, token_type_ids, image_tensor,
                return_sequence=is_seq_task
            )
            
            # 切换到历史任务的head
            if hasattr(model, 'set_active_head'):
                model.set_active_head(session_name, strict=False)
                logits = model.head(fused_feat)
            else:
                logger.warning(f"Model doesn't have set_active_head, skipping replay for {session_name}")
                return None
            
            # 计算损失
            if is_seq_task:
                # Token级分类
                logits_flat = logits.reshape(-1, logits.size(-1))
                labels_flat = labels.reshape(-1)
                
                class_weights = self._get_class_weights(task_name, device)
                loss = F.cross_entropy(
                    logits_flat,
                    labels_flat,
                    weight=class_weights,
                    ignore_index=-100
                )
            else:
                # 句级分类
                class_weights = self._get_class_weights(task_name, device)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
            
            logger.debug(f"Session '{session_name}' replay loss: {loss.item():.4f}")
            return loss
        
        except Exception as e:
            logger.error(f"Error computing replay loss for {session_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

