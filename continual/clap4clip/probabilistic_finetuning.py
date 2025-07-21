import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math

class ProbabilisticFinetuning(nn.Module):
    """
    概率微调模块
    基于论文: https://arxiv.org/abs/2403.19137
    """
    def __init__(self, 
                 model: nn.Module,
                 num_tasks: int,
                 finetune_lambda: float = 0.1,
                 temperature: float = 0.07,
                 use_uncertainty: bool = True):
        super().__init__()
        self.model = model
        self.num_tasks = num_tasks
        self.finetune_lambda = finetune_lambda
        self.temperature = temperature
        self.use_uncertainty = use_uncertainty
        
        # 任务权重参数
        self.task_weights = nn.Parameter(torch.ones(num_tasks) / num_tasks)
        
        # 不确定性估计
        if use_uncertainty:
            self.uncertainty_estimator = nn.Sequential(
                nn.Linear(model.hidden_size if hasattr(model, 'hidden_size') else 512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        
    def compute_task_probabilities(self, 
                                 text_features: torch.Tensor, 
                                 image_features: torch.Tensor,
                                 uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算任务概率分布
        Args:
            text_features: 文本特征 [batch_size, hidden_size]
            image_features: 图像特征 [batch_size, hidden_size]
            uncertainty: 不确定性估计 [batch_size, 1]
        Returns:
            task_probs: 任务概率 [batch_size, num_tasks]
        """
        batch_size = text_features.size(0)
        
        # 计算文本和图像特征的相似度
        similarity = F.cosine_similarity(text_features, image_features, dim=-1)  # [batch_size]
        
        # 基础任务权重
        base_weights = F.softmax(self.task_weights, dim=0)  # [num_tasks]
        
        # 扩展为batch维度
        task_logits = base_weights.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_tasks]
        
        # 根据相似度调整权重
        similarity_factor = similarity.unsqueeze(1) / self.temperature  # [batch_size, 1]
        task_logits = task_logits * similarity_factor
        
        # 如果使用不确定性，进一步调整
        if self.use_uncertainty and uncertainty is not None:
            uncertainty_factor = 1.0 - uncertainty  # 不确定性越高，权重越低
            task_logits = task_logits * uncertainty_factor
        
        # 计算最终概率
        task_probs = F.softmax(task_logits, dim=1)
        
        return task_probs
    
    def estimate_uncertainty(self, features: torch.Tensor) -> torch.Tensor:
        """
        估计特征的不确定性
        Args:
            features: 输入特征 [batch_size, hidden_size]
        Returns:
            uncertainty: 不确定性估计 [batch_size, 1]
        """
        if not self.use_uncertainty:
            return torch.zeros(features.size(0), 1, device=features.device)
        
        return self.uncertainty_estimator(features)
    
    def probabilistic_update(self, 
                           text_features: torch.Tensor, 
                           image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        概率微调更新
        Args:
            text_features: 文本特征 [batch_size, hidden_size]
            image_features: 图像特征 [batch_size, hidden_size]
        Returns:
            text_finetuned: 微调后的文本特征
            image_finetuned: 微调后的图像特征
            task_probs: 任务概率
        """
        # 估计不确定性
        combined_features = text_features + image_features
        uncertainty = self.estimate_uncertainty(combined_features)
        
        # 计算任务概率
        task_probs = self.compute_task_probabilities(text_features, image_features, uncertainty)
        
        # 应用概率微调
        finetune_factor = 1.0 + self.finetune_lambda * task_probs.mean(dim=1, keepdim=True)
        
        text_finetuned = text_features * finetune_factor
        image_finetuned = image_features * finetune_factor
        
        return text_finetuned, image_finetuned, task_probs
    
    def forward(self, 
                text_features: torch.Tensor, 
                image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            text_features: 文本特征
            image_features: 图像特征
        Returns:
            text_finetuned: 微调后的文本特征
            image_finetuned: 微调后的图像特征
            task_probs: 任务概率
        """
        return self.probabilistic_update(text_features, image_features)
    
    def get_task_weights(self) -> torch.Tensor:
        """获取当前任务权重"""
        return F.softmax(self.task_weights, dim=0)
    
    def update_task_weights(self, task_losses: torch.Tensor):
        """
        根据任务损失更新权重
        Args:
            task_losses: 各任务的损失 [num_tasks]
        """
        with torch.no_grad():
            # 基于损失的反向更新
            loss_weights = 1.0 / (task_losses + 1e-8)
            normalized_weights = F.softmax(loss_weights, dim=0)
            
            # 平滑更新
            alpha = 0.1
            current_weights = F.softmax(self.task_weights, dim=0)
            new_weights = alpha * normalized_weights + (1 - alpha) * current_weights
            
            # 更新参数
            self.task_weights.data = torch.log(new_weights + 1e-8)
