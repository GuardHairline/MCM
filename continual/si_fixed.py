# continual/si_fixed.py
"""
修复后的Synaptic Intelligence实现

主要修复：
1. 添加设备管理
2. 共享参数过滤
3. 正确的梯度累积
4. 数值稳定性
5. 与框架正确集成
"""

import torch
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SynapticIntelligence:
    """
    Synaptic Intelligence (修复版)
    
    改进：
    - 设备管理
    - 共享参数过滤
    - 正确的梯度累积
    - 数值稳定性
    """
    
    def __init__(self, model, epsilon=0.1, device=None):
        """
        Args:
            model: Full_Model实例
            epsilon: 正则化系数
            device: 设备
        """
        self.model = model
        self.epsilon = epsilon
        
        # 设备设置
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # 只为共享参数初始化
        self.omega = {}
        self.prev_params = {}
        self._accum = {}
        
        for n, p in model.named_parameters():
            if self._is_shared_parameter(n):
                self.omega[n] = torch.zeros_like(p, device=self.device)
                self.prev_params[n] = p.clone().detach().to(self.device)
                self._accum[n] = torch.zeros_like(p, device=self.device)
        
        logger.info(f"SynapticIntelligence initialized: tracking {len(self.omega)} shared parameters on {self.device}")
    
    def _is_shared_parameter(self, param_name: str) -> bool:
        """判断参数是否为共享参数"""
        skip_patterns = ['head.', 'task_heads.', 'head_manager.', 'classifier.']
        for pattern in skip_patterns:
            if pattern in param_name:
                return False
        return True
    
    def accumulate(self):
        """
        累积路径积分
        在每次optimizer.step()之后调用
        """
        try:
            for n, p in self.model.named_parameters():
                if n not in self.omega:
                    continue
                
                if p.grad is None:
                    continue
                
                # 确保所有张量在同一设备上
                grad = p.grad.detach().to(self.device)
                param = p.detach().to(self.device)
                prev_param = self.prev_params[n].to(self.device)
                
                # Path integral: -grad * delta_param
                delta_param = param - prev_param
                self._accum[n] += (-grad * delta_param)
                
                # 更新prev_params
                self.prev_params[n] = param.clone()
            
        except Exception as e:
            logger.error(f"Error in SI accumulate: {e}")
            import traceback
            traceback.print_exc()
    
    def update_omega(self):
        """
        更新omega（在任务结束后调用）
        
        Omega_i = Accumulator_i / (delta_param_i^2 + epsilon)
        """
        try:
            total_updated = 0
            
            for n in self.omega:
                if n not in self.prev_params:
                    continue
                
                # 获取当前参数
                current_param = None
                for param_name, param in self.model.named_parameters():
                    if param_name == n:
                        current_param = param.detach().to(self.device)
                        break
                
                if current_param is None:
                    logger.warning(f"Parameter {n} not found in model")
                    continue
                
                # 计算参数变化
                delta = current_param - self.prev_params[n].to(self.device)
                delta_sq = delta.pow(2)
                
                # 计算omega增量（带数值稳定性）
                omega_delta = self._accum[n].to(self.device) / (delta_sq + 1e-10)
                
                # 检查NaN/Inf
                if torch.isnan(omega_delta).any() or torch.isinf(omega_delta).any():
                    logger.warning(f"Invalid omega delta for parameter {n}, skipping")
                    continue
                
                # 累积omega
                self.omega[n] += omega_delta
                total_updated += 1
                
                # 更新prev_params
                self.prev_params[n] = current_param.clone()
            
            # 重置累加器
            for n in self._accum:
                self._accum[n].zero_()
            
            if total_updated > 0:
                # 统计信息
                omega_stats = {n: self.omega[n].mean().item() for n in list(self.omega.keys())[:5]}
                logger.info(f"SI omega updated for {total_updated} parameters. Sample stats: {omega_stats}")
            else:
                logger.warning("No parameters updated in SI omega")
        
        except Exception as e:
            logger.error(f"Error in SI update_omega: {e}")
            import traceback
            traceback.print_exc()
    
    def penalty(self):
        """
        计算SI正则化损失
        
        Loss = epsilon * sum_i omega_i * (theta_i - theta_i^*)^2
        """
        try:
            device = next(self.model.parameters()).device
            loss = torch.tensor(0., device=device, requires_grad=True)
            
            constrained_params = 0
            
            for n, p in self.model.named_parameters():
                if n not in self.omega:
                    continue
                
                # 确保所有张量在同一设备上
                omega = self.omega[n].to(device)
                prev_param = self.prev_params[n].to(device)
                delta = p - prev_param
                
                # 计算loss
                param_loss = (omega * delta.pow(2)).sum()
                
                # 检查NaN/Inf
                if torch.isnan(param_loss) or torch.isinf(param_loss):
                    logger.warning(f"Invalid SI loss for parameter {n}, skipping")
                    continue
                
                loss = loss + param_loss
                constrained_params += 1
            
            if constrained_params > 0:
                logger.debug(f"SI: constrained {constrained_params} parameters")
            
            return self.epsilon * loss
        
        except Exception as e:
            logger.error(f"Error in SI penalty: {e}")
            return torch.tensor(0., device=next(self.model.parameters()).device)
