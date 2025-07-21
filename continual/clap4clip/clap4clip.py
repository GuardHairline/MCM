import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel
from typing import Optional, Dict, Any
import math

class TaskAdapter(nn.Module):
    """任务特定的适配器"""
    def __init__(self, hidden_size: int, adapter_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x + residual

class CLAP4CLIP(nn.Module):
    """
    CLAP4CLIP: Continual Learning with Adapters and Probabilistic Finetuning for CLIP
    基于论文: https://arxiv.org/abs/2403.19137
    """
    def __init__(self, 
                 text_model_name: str = "openai/clip-vit-base-patch32",
                 image_model_name: str = "openai/clip-vit-base-patch32",
                 num_labels: int = 3,
                 dropout_prob: float = 0.1,
                 adapter_size: int = 64,
                 finetune_lambda: float = 0.1,
                 temperature: float = 0.07):
        super().__init__()
        
        # 使用CLIP模型（论文中的选择）
        self.clip_model = CLIPModel.from_pretrained(text_model_name)
        self.text_encoder = self.clip_model.text_model
        self.vision_encoder = self.clip_model.vision_model
        self.text_projection = self.clip_model.text_projection
        self.visual_projection = self.clip_model.visual_projection
        self.projection_dim = self.clip_model.config.projection_dim
        
        # 冻结原始CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 模型配置
        self.num_labels = num_labels
        self.adapter_size = adapter_size
        self.finetune_lambda = finetune_lambda
        self.temperature = temperature
        
        # 任务适配器（projection_dim）
        self.text_adapter = TaskAdapter(self.projection_dim, adapter_size, dropout_prob)
        self.vision_adapter = TaskAdapter(self.projection_dim, adapter_size, dropout_prob)
        
        # 分类头
        self.classifier = nn.Linear(self.projection_dim, num_labels)
        self.dropout = nn.Dropout(dropout_prob)
        
        # 任务管理
        self.task_heads = {}
        self.current_task = None
        
        # 概率微调参数
        self.task_weights = None
        self.task_probabilities = None
        
        # 设备跟踪
        self.device = None
        
    def to(self, device):
        """重写to方法，确保任务头也被移动到正确的设备"""
        super().to(device)
        self.device = device
        # 移动所有任务头到正确的设备
        for task_name, task_head in self.task_heads.items():
            self.task_heads[task_name] = task_head.to(device)
        return self
        
    def add_task(self, task_name: str, num_labels: int):
        """添加新任务"""
        task_classifier = nn.Linear(self.projection_dim, num_labels)
        # 确保任务头在正确的设备上
        if hasattr(self, 'device'):
            task_classifier = task_classifier.to(self.device)
        self.task_heads[task_name] = task_classifier
        return task_classifier
        
    def set_current_task(self, task_name: str):
        """设置当前任务"""
        if task_name not in self.task_heads:
            raise ValueError(f"Task {task_name} not found in task_heads")
        self.current_task = task_name
        
    def compute_task_probabilities(self, text_features: torch.Tensor, image_features: torch.Tensor):
        """计算任务概率分布"""
        similarity = F.cosine_similarity(text_features, image_features, dim=-1)
        task_logits = similarity.unsqueeze(1) / self.temperature
        if len(self.task_heads) > 1:
            task_probs = F.softmax(task_logits, dim=-1)
        else:
            task_probs = torch.ones_like(task_logits)
        return task_probs
        
    def probabilistic_finetuning(self, text_features: torch.Tensor, image_features: torch.Tensor):
        """概率微调过程"""
        task_probs = self.compute_task_probabilities(text_features, image_features)
        text_finetuned = text_features * (1 + self.finetune_lambda * task_probs.mean(dim=1, keepdim=True))
        image_finetuned = image_features * (1 + self.finetune_lambda * task_probs.mean(dim=1, keepdim=True))
        return text_finetuned, image_finetuned, task_probs
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                image_tensor: torch.Tensor,
                task_name: Optional[str] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            input_ids: 文本输入ID
            attention_mask: 注意力掩码
            image_tensor: 图像张量
            task_name: 任务名称（可选）
        """
        # 获取文本和图像特征（CLIP模型的标准输出）
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output  # [batch_size, text_hidden]
        text_features = self.text_projection(text_features)  # [batch_size, projection_dim]
        
        image_outputs = self.vision_encoder(image_tensor)
        image_features = image_outputs.pooler_output  # [batch_size, vision_hidden]
        image_features = self.visual_projection(image_features)  # [batch_size, projection_dim]
        
        # 应用任务适配器
        text_features = self.text_adapter(text_features)
        image_features = self.vision_adapter(image_features)
        
        # 概率微调
        text_finetuned, image_finetuned, task_probs = self.probabilistic_finetuning(
            text_features, image_features
        )
        
        # 特征融合
        combined_features = text_finetuned + image_finetuned
        combined_features = self.dropout(combined_features)
        
        # 分类
        if task_name and task_name in self.task_heads:
            logits = self.task_heads[task_name](combined_features)
        elif self.current_task and self.current_task in self.task_heads:
            logits = self.task_heads[self.current_task](combined_features)
        else:
            logits = self.classifier(combined_features)
        return logits
        
    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output
        text_features = self.text_projection(text_features)
        return self.text_adapter(text_features)
        
    def get_image_features(self, image_tensor: torch.Tensor):
        image_outputs = self.vision_encoder(image_tensor)
        image_features = image_outputs.pooler_output
        image_features = self.visual_projection(image_features)
        return self.vision_adapter(image_features)
        
    def compute_contrastive_loss(self, text_features: torch.Tensor, image_features: torch.Tensor):
        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        logits = torch.matmul(text_features, image_features.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss
