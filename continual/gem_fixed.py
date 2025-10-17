# continual/gem_fixed.py
"""
修复后的GEM实现

主要修复：
1. 使用label_config判断任务类型
2. 使用label_config获取类别权重
3. 改进head访问逻辑
4. 更好的异常处理
5. 改进梯度投影算法
"""

import os
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
import logging
from utils.ensureFileExists import ensure_directory_exists
from continual.label_config import get_label_manager

logger = logging.getLogger(__name__)


class GEMManager:
    """
    Gradient Episodic Memory (GEM) 管理器（修复版）
    
    改进：
    - 使用label_config判断任务类型
    - 正确的类别权重
    - 改进的head切换
    - 更稳定的梯度投影
    """
    
    def __init__(self, model, memory_size=100, mem_dir="gem_memory", device=None):
        """
        Args:
            model: Full_Model实例
            memory_size: 每个任务存储的记忆样本数
            mem_dir: 记忆样本保存目录
            device: 设备
        """
        self.model = model
        self.memory_size = memory_size
        self.mem_dir = mem_dir
        ensure_directory_exists(mem_dir)
        self.mem_data: Dict[str, List[dict]] = {}  # task_name -> list of samples
        
        # 设备设置
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"GEMManager initialized: memory_size={memory_size}, device={self.device}")
    
    def _is_sequence_task(self, task_name: str) -> bool:
        """判断是否为序列任务"""
        return get_label_manager().is_token_level_task(task_name)
    
    def _get_class_weights(self, task_name: str) -> Optional[torch.Tensor]:
        """获取类别权重"""
        weights = get_label_manager().get_class_weights(task_name)
        if weights is not None:
            return torch.tensor(weights, dtype=torch.float32, device=self.device)
        return None
    
    def register_task(self, task_name: str, dataset) -> None:
        """
        为新任务注册记忆样本
        
        改进：
        - 更好的异常处理
        - 更清晰的日志
        """
        mem_file = os.path.join(self.mem_dir, f"{task_name}_mem.pt")
        
        if os.path.exists(mem_file):
            try:
                self.mem_data[task_name] = torch.load(mem_file, map_location='cpu')
                logger.info(f"Loaded GEM memory for task '{task_name}' from {mem_file}")
                return
            except Exception as e:
                logger.warning(f"Failed to load GEM memory from {mem_file}: {e}")
        
        # 随机采样
        import random
        num = min(self.memory_size, len(dataset))
        indices = random.sample(range(len(dataset)), num)
        samples = [dataset[i] for i in indices]
        self.mem_data[task_name] = samples
        
        logger.info(f"Registered GEM memory for task '{task_name}': {num} samples")
    
    def save_memory(self, task_name: str) -> None:
        """将任务记忆样本保存到磁盘"""
        try:
            mem_file = os.path.join(self.mem_dir, f"{task_name}_mem.pt")
            torch.save(self.mem_data.get(task_name, []), mem_file)
            logger.info(f"Saved GEM memory for task '{task_name}' to {mem_file}")
        except Exception as e:
            logger.error(f"Failed to save GEM memory for task '{task_name}': {e}")
    
    def project_gradients(self, grads=None, current_grad=None) -> None:
        """
        梯度投影以避免灾难性遗忘
        
        改进：
        - 使用label_config
        - 正确的head切换
        - 更稳定的投影算法
        - 添加数值稳定性检查
        """
        if not self.mem_data:
            return  # 没有记忆样本，直接返回
        
        try:
            mem_grads = []
            
            # 遍历所有历史任务，计算记忆梯度
            for task_name, samples in self.mem_data.items():
                if not samples:
                    continue
                
                # 判断任务类型
                is_seq = self._is_sequence_task(task_name)
                
                # 批量化记忆样本
                batch = self._batch_collate(samples)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                image_tensor = batch['image_tensor'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.model.zero_grad()
                
                # 前向传播
                fused_feat = self.model.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    image_tensor=image_tensor,
                    return_sequence=is_seq
                )
                
                # 切换到历史任务的head
                if hasattr(self.model, 'set_active_head'):
                    try:
                        self.model.set_active_head(task_name, strict=False)
                        logits = self.model.head(fused_feat)
                    except Exception as e:
                        logger.warning(f"Failed to set head for task '{task_name}': {e}, skipping")
                        continue
                else:
                    logger.warning(f"Model doesn't have set_active_head, skipping task '{task_name}'")
                    continue
                
                # 计算损失
                class_weights = self._get_class_weights(task_name)
                if is_seq:
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        weight=class_weights,
                        ignore_index=-100
                    )
                else:
                    loss = F.cross_entropy(logits, labels, weight=class_weights)
                
                loss.backward()
                
                # 收集梯度向量（只收集共享参数的梯度）
                vec = []
                for param_name, param in self.model.named_parameters():
                    if param.grad is not None:
                        # 只约束共享参数
                        if self._is_shared_parameter(param_name):
                            vec.append(param.grad.detach().contiguous().view(-1))
                
                if vec:
                    mem_grads.append(torch.cat(vec))
            
            if not mem_grads:
                return
            
            # 计算所有记忆梯度的平均值
            mem_grad_vec = torch.stack(mem_grads).mean(dim=0)
            
            # 收集当前梯度
            cur_grad_parts = []
            for param_name, param in self.model.named_parameters():
                if param.grad is not None and self._is_shared_parameter(param_name):
                    cur_grad_parts.append(param.grad.detach().contiguous().view(-1))
            
            if not cur_grad_parts:
                return
            
            cur_grad_vec = torch.cat(cur_grad_parts)
            
            # 检查维度匹配
            if cur_grad_vec.shape != mem_grad_vec.shape:
                logger.warning(
                    f"Gradient shape mismatch: current={cur_grad_vec.shape}, "
                    f"memory={mem_grad_vec.shape}, skipping projection"
                )
                return
            
            # 计算点积
            dot_product = torch.dot(cur_grad_vec, mem_grad_vec)
            
            # 若点积为负，则投影到与记忆梯度一致的半空间
            if dot_product < 0:
                mem_grad_norm_sq = mem_grad_vec.norm() ** 2
                
                # 数值稳定性检查
                if mem_grad_norm_sq < 1e-8:
                    logger.warning("Memory gradient norm too small, skipping projection")
                    return
                
                # 计算投影
                proj = (dot_product / mem_grad_norm_sq) * mem_grad_vec
                new_vec = cur_grad_vec - proj
                
                # 将投影后的梯度写回共享参数
                pointer = 0
                for param_name, param in self.model.named_parameters():
                    if param.grad is not None and self._is_shared_parameter(param_name):
                        numel = param.grad.numel()
                        param.grad.copy_(new_vec[pointer: pointer + numel].view_as(param.grad))
                        pointer += numel
                
                logger.debug(f"GEM projection applied: dot_product={dot_product:.6f}")
            else:
                logger.debug(f"GEM projection skipped: dot_product={dot_product:.6f} >= 0")
        
        except Exception as e:
            logger.error(f"Error in GEM gradient projection: {e}")
            import traceback
            traceback.print_exc()
    
    def _is_shared_parameter(self, param_name: str) -> bool:
        """判断参数是否为共享参数"""
        # 跳过task-specific参数
        skip_patterns = ['head.', 'task_heads.', 'head_manager.', 'classifier.']
        for pattern in skip_patterns:
            if pattern in param_name:
                return False
        return True
    
    def _batch_collate(self, samples: List[dict]) -> dict:
        """将记忆样本列表组装成批量"""
        assert len(samples) > 0
        batch = {}
        for key, value in samples[0].items():
            if isinstance(value, torch.Tensor):
                batch[key] = torch.stack([s[key] for s in samples], dim=0)
        return batch

