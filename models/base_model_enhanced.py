# models/base_model_enhanced.py
"""
增强版多模态基础模型

主要改进：
1. 集成自适应融合层
2. 支持动态模态权重
3. 更灵活的融合策略
4. 保持向后兼容性
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import logging
from models.adaptive_fusion import AdaptiveFusion, DynamicModalityWeighting

logger = logging.getLogger(__name__)


class BaseMultimodalModelEnhanced(nn.Module):
    """
    增强版多模态基础模型
    
    改进：
    - 支持自适应融合（gated, attention, adaptive）
    - 动态模态权重学习
    - 更好的特征提取
    - 保持与原版base_model的兼容性
    """
    
    def __init__(self, text_model_name="microsoft/deberta-v3-base",
                 image_model_name="google/vit-base-patch16-224-in21k",
                 hidden_dim=768,
                 multimodal_fusion="gated",  # 默认使用门控融合
                 num_heads=8,
                 mode="multimodal",
                 dropout_prob=0.1,
                 use_dynamic_weighting=False):  # 是否使用动态权重
        """
        Args:
            text_model_name: 文本编码器
            image_model_name: 图像编码器
            hidden_dim: 隐藏层维度
            multimodal_fusion: 融合策略 ["gated", "attention", "concat", "add", "adaptive"]
            num_heads: 注意力头数
            mode: "text_only" 或 "multimodal"
            dropout_prob: Dropout概率
            use_dynamic_weighting: 是否使用动态模态权重
        """
        super().__init__()
        
        # 设置模型路径
        if text_model_name == "microsoft/deberta-v3-base":
            model_path = "downloaded_model/deberta-v3-base"
        elif text_model_name == "bert-base-uncased":
            model_path = "bert-base-uncased"
        else:
            model_path = text_model_name
        
        if image_model_name == "google/vit-base-patch16-224-in21k":
            image_model_path = "downloaded_model/vit-base-patch16-224-in21k"
        elif image_model_name == "resnet18":
            image_model_path = "resnet18"
        else:
            image_model_path = image_model_name
        
        # 文本编码器
        self.text_encoder = AutoModel.from_pretrained(model_path)
        self.text_hidden_size = self.text_encoder.config.hidden_size
        
        # 图像编码器
        self.image_encoder = AutoModel.from_pretrained(image_model_path)
        self.image_hidden_size = self.image_encoder.config.hidden_size
        
        # 图像特征投影
        self.image_proj = nn.Linear(self.image_hidden_size, self.text_hidden_size)
        
        # 融合策略
        self.fusion_strategy = multimodal_fusion
        self.mode = mode
        self.use_dynamic_weighting = use_dynamic_weighting
        
        # 创建自适应融合层
        if mode == "multimodal":
            self.fusion = AdaptiveFusion(
                hidden_dim=self.text_hidden_size,
                fusion_type=multimodal_fusion,
                num_heads=num_heads,
                dropout_prob=dropout_prob
            )
            
            # 动态模态权重（可选）
            if use_dynamic_weighting:
                self.modality_weighting = DynamicModalityWeighting(self.text_hidden_size)
        
        self.fusion_output_dim = self.text_hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        
        logger.info(
            f"BaseMultimodalModelEnhanced initialized: "
            f"fusion={multimodal_fusion}, dynamic_weighting={use_dynamic_weighting}"
        )
    
    def get_image_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            image_tensor: (batch_size, C, H, W)
        
        Returns:
            image_feat: (batch_size, hidden_dim)
        """
        outputs = self.image_encoder(image_tensor)
        features = outputs.last_hidden_state[:, 0, :]  # 使用CLS token
        return self.image_proj(features)
    
    def forward(self, input_ids, attention_mask, token_type_ids, image_tensor, 
                return_sequence=False):
        """
        前向传播
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len) 或 None
            image_tensor: (batch_size, C, H, W)
            return_sequence: 是否返回序列特征
        
        Returns:
            如果return_sequence=True: (batch_size, seq_len, hidden_dim)
            如果return_sequence=False: (batch_size, hidden_dim)
        """
        # 1. 文本特征
        if token_type_ids is not None:
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        text_sequence = text_outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        text_cls = text_sequence[:, 0, :]  # (batch_size, hidden_dim)
        
        # 如果是纯文本模式
        if self.mode == "text_only":
            if return_sequence:
                return text_sequence
            else:
                return text_cls
        
        # 2. 图像特征
        image_feat = self.get_image_features(image_tensor)  # (batch_size, hidden_dim)
        
        # 3. 多模态融合
        if return_sequence:
            # 序列级融合
            fused_feat = self.fusion(text_sequence, image_feat)
        else:
            # 句子级融合
            if self.use_dynamic_weighting:
                # 使用动态权重
                modality_weights, fused_feat = self.modality_weighting(text_cls, image_feat)
                # 可以记录权重用于分析
                if hasattr(self, '_last_modality_weights'):
                    self._last_modality_weights = modality_weights
            else:
                # 使用自适应融合
                fused_feat = self.fusion(text_cls, image_feat)
        
        return fused_feat
    
    def get_last_modality_weights(self):
        """
        获取最后一次前向传播的模态权重（如果使用动态权重）
        
        Returns:
            weights: (batch_size, 2) [text_weight, image_weight] 或 None
        """
        if self.use_dynamic_weighting and hasattr(self, '_last_modality_weights'):
            return self._last_modality_weights
        return None

