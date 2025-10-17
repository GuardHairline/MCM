# models/adaptive_fusion.py
"""
自适应多模态融合层

主要特性：
1. 门控融合（Gated Fusion）- 动态调整模态权重
2. 注意力融合（Attention Fusion）- 基于注意力的模态融合
3. 分层融合（Hierarchical Fusion）- 支持早期融合和晚期融合
4. 可学习的模态权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GatedFusion(nn.Module):
    """
    门控融合模块
    
    使用门控机制动态调整文本和图像模态的权重
    """
    
    def __init__(self, hidden_dim: int, dropout_prob: float = 0.1):
        """
        Args:
            hidden_dim: 隐藏层维度
            dropout_prob: Dropout概率
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 门控网络
        self.gate_text = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.gate_image = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 融合后的投影
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_feat: (batch_size, [seq_len], hidden_dim) 文本特征
            image_feat: (batch_size, hidden_dim) 图像特征
        
        Returns:
            fused_feat: (batch_size, [seq_len], hidden_dim) 融合后特征
        """
        # 扩展图像特征维度（如果文本是序列）
        if text_feat.dim() == 3:
            # 序列特征
            image_feat = image_feat.unsqueeze(1).expand(-1, text_feat.size(1), -1)
        
        # 计算门控值
        gate_t = self.gate_text(text_feat)  # (batch_size, [seq_len], 1)
        gate_i = self.gate_image(image_feat)  # (batch_size, [seq_len], 1)
        
        # 归一化门控值
        gate_sum = gate_t + gate_i + 1e-8
        gate_t_norm = gate_t / gate_sum
        gate_i_norm = gate_i / gate_sum
        
        # 加权融合
        fused = gate_t_norm * text_feat + gate_i_norm * image_feat
        
        # 投影
        fused = self.fusion_proj(fused)
        
        return fused


class AttentionFusion(nn.Module):
    """
    基于注意力的融合模块
    
    使用cross-attention让文本特征attend到图像特征
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout_prob: float = 0.1):
        """
        Args:
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout_prob: Dropout概率
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Cross-attention: text as query, image as key/value
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True
        )
        
        # Layer norm
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_feat: (batch_size, seq_len, hidden_dim) 或 (batch_size, hidden_dim)
            image_feat: (batch_size, hidden_dim)
        
        Returns:
            fused_feat: 与text_feat相同shape
        """
        # 确保text_feat是3D的
        is_2d = (text_feat.dim() == 2)
        if is_2d:
            text_feat = text_feat.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # 图像特征也扩展为3D
        image_feat = image_feat.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Cross-attention
        attn_output, attn_weights = self.cross_attention(
            query=text_feat,
            key=image_feat,
            value=image_feat
        )
        
        # Residual connection + layer norm
        text_feat = self.layer_norm1(text_feat + attn_output)
        
        # FFN
        ffn_output = self.ffn(text_feat)
        text_feat = self.layer_norm2(text_feat + ffn_output)
        
        # 如果输入是2D，输出也是2D
        if is_2d:
            text_feat = text_feat.squeeze(1)
        
        return text_feat


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块
    
    根据任务和输入动态选择最佳融合策略
    """
    
    def __init__(self, hidden_dim: int, fusion_type: str = "gated",
                 num_heads: int = 8, dropout_prob: float = 0.1):
        """
        Args:
            hidden_dim: 隐藏层维度
            fusion_type: 融合类型 ["gated", "attention", "concat", "add", "adaptive"]
            num_heads: 注意力头数（用于attention融合）
            dropout_prob: Dropout概率
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        
        if fusion_type == "gated":
            self.fusion = GatedFusion(hidden_dim, dropout_prob)
        elif fusion_type == "attention":
            self.fusion = AttentionFusion(hidden_dim, num_heads, dropout_prob)
        elif fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_prob)
            )
        elif fusion_type == "add":
            self.fusion = nn.Identity()
        elif fusion_type == "adaptive":
            # 自适应：同时使用门控和注意力，学习如何组合
            self.gated_fusion = GatedFusion(hidden_dim, dropout_prob)
            self.attention_fusion = AttentionFusion(hidden_dim, num_heads, dropout_prob)
            self.strategy_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        logger.info(f"AdaptiveFusion initialized with type='{fusion_type}'")
    
    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_feat: (batch_size, [seq_len], hidden_dim) 文本特征
            image_feat: (batch_size, hidden_dim) 图像特征
        
        Returns:
            fused_feat: 与text_feat相同shape的融合特征
        """
        if self.fusion_type == "gated":
            return self.fusion(text_feat, image_feat)
        
        elif self.fusion_type == "attention":
            return self.fusion(text_feat, image_feat)
        
        elif self.fusion_type == "concat":
            # 扩展图像特征
            if text_feat.dim() == 3:
                image_feat = image_feat.unsqueeze(1).expand(-1, text_feat.size(1), -1)
            # 拼接
            concat_feat = torch.cat([text_feat, image_feat], dim=-1)
            return self.fusion(concat_feat)
        
        elif self.fusion_type == "add":
            # 扩展图像特征
            if text_feat.dim() == 3:
                image_feat = image_feat.unsqueeze(1).expand(-1, text_feat.size(1), -1)
            return text_feat + image_feat
        
        elif self.fusion_type == "adaptive":
            # 组合门控和注意力
            gated_out = self.gated_fusion(text_feat, image_feat)
            attn_out = self.attention_fusion(text_feat, image_feat)
            
            # 归一化权重
            weights = F.softmax(self.strategy_weight, dim=0)
            
            return weights[0] * gated_out + weights[1] * attn_out
        
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")


class DynamicModalityWeighting(nn.Module):
    """
    动态模态权重模块
    
    根据输入内容学习模态的重要性权重
    """
    
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 权重预测网络
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 输出2个权重：[text_weight, image_weight]
            nn.Softmax(dim=-1)
        )
    
    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_feat: (batch_size, hidden_dim) 文本特征
            image_feat: (batch_size, hidden_dim) 图像特征
        
        Returns:
            weights: (batch_size, 2) 模态权重 [text_weight, image_weight]
            fused_feat: (batch_size, hidden_dim) 加权融合特征
        """
        # 拼接特征
        concat_feat = torch.cat([text_feat, image_feat], dim=-1)
        
        # 预测权重
        weights = self.weight_predictor(concat_feat)  # (batch_size, 2)
        
        # 加权融合
        text_weight = weights[:, 0:1]  # (batch_size, 1)
        image_weight = weights[:, 1:2]  # (batch_size, 1)
        
        fused_feat = text_weight * text_feat + image_weight * image_feat
        
        return weights, fused_feat

