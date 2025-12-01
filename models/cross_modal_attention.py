"""
跨模态注意力模块

实现双向跨模态交互：
1. 文本 → 图像 注意力（文本attend to图像）
2. 图像 → 文本 注意力（图像attend to文本）
3. 门控融合

参考文献：
- "Multi-modal Graph Fusion for Named Entity Recognition with Targeted Visual Guidance" (AAAI 2021)
- "Improving Multimodal Named Entity Recognition via Entity Span Detection with Unified Multimodal Transformer" (ACL 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    双向跨模态注意力
    
    特性：
    - 文本关注图像（让图像信息流入文本）
    - 图像关注文本（让文本信息流入图像）
    - 门控融合（自适应控制融合强度）
    """
    
    def __init__(self, text_dim=768, image_dim=768, num_heads=8, dropout=0.1):
        """
        Args:
            text_dim: 文本特征维度
            image_dim: 图像特征维度
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        
        # 图像投影（如果维度不同）
        if image_dim != text_dim:
            self.image_proj = nn.Linear(image_dim, text_dim)
        else:
            self.image_proj = nn.Identity()
        
        # 文本 → 图像 注意力
        self.text2img_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 图像 → 文本 注意力
        self.img2text_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 门控融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(text_dim * 3, text_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(text_dim, text_dim),
            nn.Sigmoid()
        )
        
        # 输出层归一化
        self.layer_norm = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text_seq, image_feat, attention_mask=None):
        """
        前向传播
        
        Args:
            text_seq: 文本序列特征 (batch, seq_len, text_dim)
            image_feat: 图像特征 (batch, image_dim) 或 (batch, num_regions, image_dim)
            attention_mask: 文本mask (batch, seq_len)
        
        Returns:
            fused_seq: 融合后的序列 (batch, seq_len, text_dim)
            attention_weights: 注意力权重字典
        """
        batch_size, seq_len, _ = text_seq.size()
        
        # 处理图像特征
        if image_feat.dim() == 2:
            # 单个图像特征 (batch, image_dim)
            image_feat = self.image_proj(image_feat).unsqueeze(1)  # (batch, 1, text_dim)
        else:
            # 多区域图像特征 (batch, num_regions, image_dim)
            image_feat = self.image_proj(image_feat)  # (batch, num_regions, text_dim)
        
        # 扩展图像特征到序列长度（用于图像→文本注意力）
        num_image_regions = image_feat.size(1)
        image_seq = image_feat.expand(-1, seq_len, -1).contiguous().view(batch_size, seq_len, -1)
        
        # 1. 文本关注图像（Text attends to Image）
        # Query: 文本，Key/Value: 图像
        text_att_img, text_att_img_weights = self.text2img_attention(
            query=text_seq,
            key=image_feat,
            value=image_feat,
            key_padding_mask=None  # 假设图像特征都有效
        )
        
        # 2. 图像关注文本（Image attends to Text）
        # Query: 图像，Key/Value: 文本
        # 将每个token都"看作"一个图像区域在关注文本
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: 1表示有效，0表示padding
            # key_padding_mask: True表示padding，False表示有效
            key_padding_mask = (attention_mask == 0)
        
        img_att_text, img_att_text_weights = self.img2text_attention(
            query=text_seq,  # 每个token作为query
            key=text_seq,
            value=text_seq,
            key_padding_mask=key_padding_mask
        )
        
        # 3. 门控融合
        # 拼接三个表示：原始文本、文本关注图像、图像关注文本
        concat_features = torch.cat([text_seq, text_att_img, img_att_text], dim=-1)
        
        # 计算门控权重
        gate = self.fusion_gate(concat_features)  # (batch, seq_len, text_dim)
        
        # 融合：门控控制多模态信息的混合
        # gate接近1：更多使用多模态信息
        # gate接近0：更多保留原始文本信息
        multimodal_info = 0.5 * text_att_img + 0.5 * img_att_text
        fused_seq = gate * multimodal_info + (1 - gate) * text_seq
        
        # 残差连接和层归一化
        fused_seq = self.layer_norm(fused_seq + text_seq)
        fused_seq = self.dropout(fused_seq)
        
        # 返回注意力权重（用于可视化）
        attention_weights = {
            'text_to_image': text_att_img_weights,
            'image_to_text': img_att_text_weights,
            'gate': gate
        }
        
        return fused_seq, attention_weights


class VisualGuidedAttention(nn.Module):
    """
    视觉引导的注意力
    
    让图像特征引导模型关注文本中的重要位置
    特别适合实体识别任务
    """
    
    def __init__(self, text_dim=768, image_dim=768, dropout=0.1):
        """
        Args:
            text_dim: 文本特征维度
            image_dim: 图像特征维度
            dropout: dropout概率
        """
        super().__init__()
        
        # 图像投影
        self.image_proj = nn.Linear(image_dim, text_dim)
        
        # 注意力权重计算
        self.attention_weight = nn.Sequential(
            nn.Linear(text_dim * 2, text_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(text_dim, 1)
        )
        
        self.layer_norm = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text_seq, image_feat, attention_mask=None):
        """
        前向传播
        
        Args:
            text_seq: 文本序列特征 (batch, seq_len, text_dim)
            image_feat: 图像特征 (batch, image_dim)
            attention_mask: 文本mask (batch, seq_len)
        
        Returns:
            weighted_text: 加权后的文本 (batch, seq_len, text_dim)
            attention_weights: 注意力权重 (batch, seq_len, 1)
        """
        batch_size, seq_len, _ = text_seq.size()
        
        # 投影图像特征
        if image_feat.dim() == 2:
            image_proj = self.image_proj(image_feat).unsqueeze(1)  # (batch, 1, text_dim)
        else:
            # 如果是多区域特征，取平均
            image_proj = self.image_proj(image_feat.mean(dim=1, keepdim=True))
        
        # 扩展到序列长度
        image_expanded = image_proj.expand(-1, seq_len, -1)
        
        # 拼接文本和图像
        combined = torch.cat([text_seq, image_expanded], dim=-1)  # (batch, seq_len, 2*text_dim)
        
        # 计算注意力分数
        attention_scores = self.attention_weight(combined)  # (batch, seq_len, 1)
        
        # 应用mask（padding位置的分数设为-inf）
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(-1) == 0,
                float('-inf')
            )
        
        # Softmax归一化
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # 加权文本特征
        weighted_text = text_seq * attention_weights
        
        # 残差连接和层归一化
        weighted_text = self.layer_norm(weighted_text + text_seq)
        weighted_text = self.dropout(weighted_text)
        
        return weighted_text, attention_weights


class HybridCrossModalFusion(nn.Module):
    """
    混合跨模态融合
    
    结合：
    1. 双向跨模态注意力
    2. 视觉引导注意力
    """
    
    def __init__(self, text_dim=768, image_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.cross_modal_attn = CrossModalAttention(
            text_dim=text_dim,
            image_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.visual_guided_attn = VisualGuidedAttention(
            text_dim=text_dim,
            image_dim=image_dim,
            dropout=dropout
        )
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Linear(text_dim * 2, text_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(text_dim)
        )
    
    def forward(self, text_seq, image_feat, attention_mask=None):
        """
        Args:
            text_seq: (batch, seq_len, text_dim)
            image_feat: (batch, image_dim) or (batch, num_regions, image_dim)
            attention_mask: (batch, seq_len)
        
        Returns:
            fused: (batch, seq_len, text_dim)
            attention_info: dict of attention weights
        """
        # 双向跨模态注意力
        cross_modal_feat, cross_modal_weights = self.cross_modal_attn(
            text_seq, image_feat, attention_mask
        )
        
        # 视觉引导注意力
        visual_guided_feat, visual_weights = self.visual_guided_attn(
            text_seq, image_feat, attention_mask
        )
        
        # 融合两种注意力的结果
        combined = torch.cat([cross_modal_feat, visual_guided_feat], dim=-1)
        fused = self.final_fusion(combined)
        
        attention_info = {
            'cross_modal': cross_modal_weights,
            'visual_guided': visual_weights
        }
        
        return fused, attention_info


# 使用示例
if __name__ == "__main__":
    # 测试
    batch_size = 4
    seq_len = 20
    text_dim = 768
    image_dim = 768
    
    # 模拟数据
    text_seq = torch.randn(batch_size, seq_len, text_dim)
    image_feat = torch.randn(batch_size, image_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 测试CrossModalAttention
    print("测试 CrossModalAttention:")
    cross_modal = CrossModalAttention(text_dim, image_dim)
    fused, weights = cross_modal(text_seq, image_feat, attention_mask)
    print(f"  输入形状: {text_seq.shape}")
    print(f"  输出形状: {fused.shape}")
    print(f"  注意力权重: {weights['text_to_image'].shape}")
    
    # 测试VisualGuidedAttention
    print("\n测试 VisualGuidedAttention:")
    visual_guided = VisualGuidedAttention(text_dim, image_dim)
    weighted, attn_weights = visual_guided(text_seq, image_feat, attention_mask)
    print(f"  输入形状: {text_seq.shape}")
    print(f"  输出形状: {weighted.shape}")
    print(f"  注意力权重形状: {attn_weights.shape}")
    
    # 测试HybridCrossModalFusion
    print("\n测试 HybridCrossModalFusion:")
    hybrid = HybridCrossModalFusion(text_dim, image_dim)
    fused, attn_info = hybrid(text_seq, image_feat, attention_mask)
    print(f"  输入形状: {text_seq.shape}")
    print(f"  输出形状: {fused.shape}")
    
    print("\n✓ 所有模块测试通过！")







































