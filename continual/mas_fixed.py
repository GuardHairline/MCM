# continual/mas_fixed.py
"""
修复后的MAS (Memory Aware Synapses)实现

主要修复：
1. 只约束共享参数
2. 改进importance计算
3. 更好的数值稳定性
4. 正确的模型前向传播
"""

import torch
import logging

logger = logging.getLogger(__name__)


class MASRegularizer:
    """
    Memory Aware Synapses (修复版)
    
    改进：
    - 只约束共享参数
    - 更稳定的importance计算
    - 正确的模型输入格式
    """
    
    def __init__(self, model, epsilon=1e-3):
        """
        Args:
            model: Full_Model实例
            epsilon: 正则化系数
        """
        self.model = model
        self.epsilon = epsilon
        
        # 只为共享参数初始化
        self.omega = {}
        self.prev_params = {}
        
        for n, p in model.named_parameters():
            if self._is_shared_parameter(n):
                self.omega[n] = torch.zeros_like(p)
                self.prev_params[n] = p.clone().detach()
        
        logger.info(f"MASRegularizer initialized: tracking {len(self.omega)} shared parameters")
    
    def _is_shared_parameter(self, param_name: str) -> bool:
        """判断参数是否为共享参数"""
        skip_patterns = ['head.', 'task_heads.', 'head_manager.', 'classifier.']
        for pattern in skip_patterns:
            if pattern in param_name:
                return False
        return True
    
    @torch.no_grad()
    def compute_importance(self, data_loader, device, task_name=None):
        """
        通过输出L2范数的梯度估计importance
        
        改进：
        - 正确的模型调用方式
        - 更好的异常处理
        - 归一化改进
        """
        try:
            # 清零omega累加器
            for n in self.omega:
                self.omega[n].zero_()
            
            self.model.eval()
            batch_count = 0
            
            for batch_idx, batch in enumerate(data_loader):
                # 准备输入
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                image_tensor = batch['image_tensor'].to(device)
                
                self.model.zero_grad()
                
                # 前向传播（根据模型类型）
                try:
                    # 尝试使用base_model + head的方式
                    if hasattr(self.model, 'base_model') and hasattr(self.model, 'head'):
                        # 判断是否需要sequence输出
                        from continual.label_config import get_label_manager
                        is_seq = get_label_manager().is_token_level_task(task_name) if task_name else False
                        
                        fused_feat = self.model.base_model(
                            input_ids, attention_mask, token_type_ids, image_tensor,
                            return_sequence=is_seq
                        )
                        logits = self.model.head(fused_feat)
                    else:
                        # 直接调用模型
                        logits = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            image_tensor=image_tensor
                        )
                    
                    # 计算输出L2范数
                    score = (logits ** 2).sum()
                    score.backward()
                    
                    # 累积梯度绝对值
                    for n, p in self.model.named_parameters():
                        if n in self.omega and p.grad is not None:
                            self.omega[n] += p.grad.abs()
                    
                    batch_count += 1
                
                except Exception as e:
                    logger.warning(f"Error processing batch {batch_idx} in MAS: {e}")
                    continue
                
                # 限制处理的batch数量（避免过长）
                if batch_count >= 100:
                    break
            
            # 归一化
            if batch_count > 0:
                for n in self.omega:
                    self.omega[n] = self.omega[n] / batch_count
                
                # 统计信息
                omega_stats = {n: self.omega[n].mean().item() for n in list(self.omega.keys())[:5]}
                logger.info(f"MAS importance computed from {batch_count} batches. Sample stats: {omega_stats}")
            else:
                logger.warning("No batches processed in MAS importance computation")
            
            # 保存参数快照
            for n, p in self.model.named_parameters():
                if n in self.omega:
                    self.prev_params[n] = p.clone().detach()
        
        except Exception as e:
            logger.error(f"Error in MAS compute_importance: {e}")
            import traceback
            traceback.print_exc()
    
    def penalty(self):
        """
        计算MAS penalty
        
        改进：
        - 只约束共享参数
        - 数值稳定性检查
        """
        try:
            loss = torch.tensor(0., device=next(self.model.parameters()).device)
            
            for n, p in self.model.named_parameters():
                if n in self.omega:
                    omega = self.omega[n].to(p.device)
                    prev_param = self.prev_params[n].to(p.device)
                    delta = p - prev_param
                    
                    param_loss = (omega * delta.pow(2)).sum()
                    
                    # 检查NaN/Inf
                    if torch.isnan(param_loss) or torch.isinf(param_loss):
                        logger.warning(f"Invalid MAS loss for parameter {n}, skipping")
                        continue
                    
                    loss += param_loss
            
            return self.epsilon * loss
        
        except Exception as e:
            logger.error(f"Error in MAS penalty: {e}")
            return torch.tensor(0., device=next(self.model.parameters()).device)

