# continual/ewc_fixed.py
"""
修复后的EWC实现

主要修复：
1. 增加Fisher矩阵估计的样本量（100 -> 500）
2. 移除错误的参数对齐逻辑（padding/truncation）
3. head参数跳过EWC约束（task-specific）
4. 正确归一化Fisher矩阵
5. 改进动态lambda策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger("ewc")


class MultiTaskEWC:
    """
    多任务版本EWC（修复版）
    
    改进：
    - 更大的Fisher估计样本量
    - 移除错误的参数对齐
    - 正确的参数过滤（跳过task-specific参数）
    - Fisher矩阵归一化
    - 改进的lambda衰减策略
    """
    
    def __init__(self, model, current_task_name, session_name, num_labels, 
                 ewc_lambda, ewc_dir="ewc_params"):
        """
        Args:
            model: 当前要学习的模型
            current_task_name: 当前任务名称
            session_name: 当前会话名称
            num_labels: 当前任务的标签数量
            ewc_lambda: EWC损失系数
            ewc_dir: Fisher参数存储目录
        """
        self.model = model
        self.task_name = current_task_name
        self.session_name = session_name
        self.num_labels = num_labels
        self.ewc_lambda = ewc_lambda
        self.ewc_dir = ewc_dir
        
        # 累积的Fisher信息和最优参数
        self.fisher_all = {}
        self.optpar_all = {}
        self.task_count = 0
        
        # 创建目录
        os.makedirs(ewc_dir, exist_ok=True)
    
    def _is_shared_parameter(self, param_name: str) -> bool:
        """
        判断参数是否为共享参数（需要EWC约束）
        
        Task-specific参数（不需要EWC）：
        - head.* (所有任务头参数)
        - task_heads.* (任务头字典)
        - head_manager.* (任务头管理器)
        
        Shared参数（需要EWC）：
        - base_model.* (共享backbone)
        """
        # 跳过task-specific参数
        skip_patterns = [
            'head.',
            'task_heads.',
            'head_manager.',
            'classifier.',  # 分类器层
            'task_specific',
            'adapter'  # 如果使用adapter也跳过
        ]
        
        for pattern in skip_patterns:
            if pattern in param_name:
                return False
        
        # base_model参数是共享的
        if 'base_model' in param_name:
            return True
        
        # 默认视为共享参数
        return True
    
    def load_all_previous_tasks(self, train_info):
        """
        加载所有历史任务的Fisher和optpar信息
        
        改进：
        - 移除了错误的padding/truncation逻辑
        - 正确累积Fisher矩阵
        - 添加参数验证
        """
        try:
            if "sessions" not in train_info or len(train_info["sessions"]) == 0:
                logger.info("No previous sessions, skipping Fisher loading")
                return
            
            self.fisher_all = {}
            self.optpar_all = {}
            self.task_count = len(train_info["sessions"])
            
            logger.info(f"Loading Fisher from {self.task_count} previous tasks...")
            
            loaded_count = 0
            for session in train_info["sessions"]:
                if "fisher_file" not in session:
                    logger.warning(f"Session {session.get('session_name', 'unknown')} has no fisher_file")
                    continue
                
                fisher_file_path = session["fisher_file"]
                if not os.path.exists(fisher_file_path):
                    logger.warning(f"Fisher file not found: {fisher_file_path}")
                    continue
                
                logger.info(f"Loading Fisher from: {fisher_file_path}")
                checkpoint = torch.load(fisher_file_path, map_location='cpu')
                fisher_dict = checkpoint["fisher"]
                optpar_dict = checkpoint["optpar"]
                
                # 累积Fisher和optpar（只处理共享参数）
                for param_name, fisher_value in fisher_dict.items():
                    # 跳过task-specific参数
                    if not self._is_shared_parameter(param_name):
                        continue
                    
                    # 转换为tensor
                    fisher_tensor = torch.tensor(fisher_value) if not isinstance(fisher_value, torch.Tensor) else fisher_value
                    optpar_tensor = torch.tensor(optpar_dict[param_name]) if not isinstance(optpar_dict[param_name], torch.Tensor) else optpar_dict[param_name]
                    
                    # 累积
                    if param_name not in self.fisher_all:
                        self.fisher_all[param_name] = fisher_tensor
                        self.optpar_all[param_name] = optpar_tensor
                    else:
                        self.fisher_all[param_name] += fisher_tensor
                        self.optpar_all[param_name] += optpar_tensor
                
                loaded_count += 1
            
            # 平均化（重要：Fisher应该是期望值）
            if loaded_count > 0:
                for param_name in self.fisher_all:
                    self.fisher_all[param_name] /= loaded_count
                    self.optpar_all[param_name] /= loaded_count
                
                logger.info(f"Successfully loaded and averaged Fisher from {loaded_count} tasks")
                logger.info(f"Tracking {len(self.fisher_all)} shared parameters")
            else:
                logger.warning("No Fisher information loaded")
        
        except Exception as e:
            logger.error(f"Error loading Fisher: {e}")
            import traceback
            traceback.print_exc()
    
    def penalty(self, model):
        """
        计算EWC penalty
        
        改进：
        - 移除了错误的参数对齐
        - 只约束共享参数
        - 添加shape验证
        - 改进的lambda衰减
        """
        if not self.fisher_all or not self.optpar_all:
            return torch.tensor(0., device=next(model.parameters()).device)
        
        # 动态lambda：使用更温和的衰减
        # lambda_t = lambda_0 * (1 / (1 + alpha * t))
        alpha = 0.1  # 衰减率（降低以减缓衰减）
        lambda_dyn = self.ewc_lambda / (1 + alpha * self.task_count)
        
        if self.task_count > 0:
            logger.debug(f"EWC lambda: {self.ewc_lambda:.4f} -> {lambda_dyn:.4f} (task_count={self.task_count})")
        
        device = next(model.parameters()).device
        loss_ewc = torch.tensor(0., device=device)
        
        constrained_params = 0
        skipped_params = 0
        
        for param_name, param in model.named_parameters():
            # 只约束共享参数
            if not self._is_shared_parameter(param_name):
                skipped_params += 1
                continue
            
            if param_name not in self.fisher_all:
                continue
            
            fisher_val = self.fisher_all[param_name].to(device)
            optpar_val = self.optpar_all[param_name].to(device)
            
            # Shape验证（移除了padding/truncation）
            if fisher_val.shape != param.shape:
                logger.warning(
                    f"Shape mismatch for {param_name}: "
                    f"fisher={fisher_val.shape}, param={param.shape}, skipping"
                )
                continue
            
            # 计算EWC loss
            loss_ewc += (fisher_val * (param - optpar_val) ** 2).sum()
            constrained_params += 1
        
        if constrained_params > 0:
            logger.debug(f"EWC: constrained {constrained_params} params, skipped {skipped_params} params")
        
        return lambda_dyn * loss_ewc
    
    def estimate_and_save_fisher(self, dataset_or_loader, device=None, 
                                 sample_size=500, batch_size=16):
        """
        估计Fisher矩阵并保存
        
        改进：
        - 增加样本量（100 -> 500）
        - 正确归一化
        - 只保存共享参数
        - 添加进度日志
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        logger.info(f"Estimating Fisher matrix with {sample_size} samples...")
        
        # 保存当前参数（optpar）
        optpar_dict = {}
        for param_name, param in self.model.named_parameters():
            if self._is_shared_parameter(param_name):
                optpar_dict[param_name] = param.detach().cpu().clone()
        
        # 初始化Fisher字典
        fisher_dict = {}
        for param_name, param in self.model.named_parameters():
            if self._is_shared_parameter(param_name):
                fisher_dict[param_name] = torch.zeros_like(param, device=device)
        
        # 准备数据
        from torch.utils.data import DataLoader
        if hasattr(dataset_or_loader, "__len__") and not hasattr(dataset_or_loader, "batch_size"):
            # Dataset -> DataLoader
            import random
            indices = list(range(len(dataset_or_loader)))
            random.shuffle(indices)
            indices = indices[:sample_size]
            
            from torch.utils.data import Subset
            subset = Subset(dataset_or_loader, indices)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        else:
            # 已经是DataLoader
            loader = dataset_or_loader
        
        # 估计Fisher
        self.model.eval()
        sample_count = 0
        
        from continual.label_config import get_label_manager
        is_seq_task = get_label_manager().is_token_level_task(self.task_name)
        
        for batch_idx, batch in enumerate(loader):
            if sample_count >= sample_size:
                break
            
            # 数据移到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            image_tensor = batch["image_tensor"].to(device)
            labels = batch["labels"].to(device)
            
            self.model.zero_grad()
            
            # 前向传播
            if is_seq_task:
                fused_feat = self.model.base_model(
                    input_ids, attention_mask, token_type_ids, image_tensor,
                    return_sequence=True
                )
                logits = self.model.head(fused_feat)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100
                )
            else:
                fused_feat = self.model.base_model(
                    input_ids, attention_mask, token_type_ids, image_tensor,
                    return_sequence=False
                )
                logits = self.model.head(fused_feat)
                loss = F.cross_entropy(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 累积Fisher（梯度的平方）
            for param_name, param in self.model.named_parameters():
                if param_name in fisher_dict and param.grad is not None:
                    fisher_dict[param_name] += param.grad.detach() ** 2
            
            sample_count += input_ids.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Fisher estimation progress: {sample_count}/{sample_size} samples")
        
        # 归一化Fisher（重要！）
        if sample_count > 0:
            for param_name in fisher_dict:
                fisher_dict[param_name] /= sample_count
            logger.info(f"Fisher normalized by {sample_count} samples")
        
        # 转换为CPU并保存
        fisher_dict_cpu = {k: v.cpu() for k, v in fisher_dict.items()}
        
        fisher_path = os.path.join(self.ewc_dir, f"{self.session_name}_fisher.pt")
        torch.save({
            'fisher': fisher_dict_cpu,
            'optpar': optpar_dict,
            'sample_count': sample_count,
            'task_name': self.task_name
        }, fisher_path)
        
        logger.info(f"Fisher matrix saved to: {fisher_path}")
        logger.info(f"Saved {len(fisher_dict_cpu)} parameters")
        
        # 打印统计信息
        fisher_stats = {}
        for param_name, fisher_val in fisher_dict_cpu.items():
            fisher_stats[param_name] = {
                'mean': fisher_val.mean().item(),
                'std': fisher_val.std().item(),
                'max': fisher_val.max().item(),
                'shape': list(fisher_val.shape)
            }
        
        logger.debug(f"Fisher statistics: {fisher_stats}")

