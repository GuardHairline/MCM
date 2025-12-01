# models/fusion/cross_modal_attention.py
"""
跨模态注意力融合模块

基于早期MNER论文(Zhang et al., AAAI 2018)的Co-Attention机制。
允许文本和图像特征进行双向交互，学习"何时看图像"。

预期：相比简单拼接，F1可提升10-15个百分点。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionFusion(nn.Module):
    """
    跨模态注意力融合
    
    实现三种融合策略：
    1. Co-Attention: 文本→图像和图像→文本的双向注意力
    2. Gated Fusion: 学习一个门控信号来平衡文本和图像
    3. Adaptive Fusion: 自适应地为每个token选择融合权重
    
    参数:
        text_dim: 文本特征维度
        img_dim: 图像特征维度
        fusion_dim: 融合后的特征维度（默认与text_dim相同）
        num_heads: 多头注意力的头数（默认8）
        fusion_type: 融合类型 ('coattn', 'gate', 'adaptive')
        dropout: Dropout概率
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        img_dim: int = 768,
        fusion_dim: int = None,
        num_heads: int = 8,
        fusion_type: str = 'coattn',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.img_dim = img_dim
        self.fusion_dim = fusion_dim or text_dim
        self.num_heads = num_heads
        self.fusion_type = fusion_type
        
        # 投影层（统一维度）
        self.text_proj = nn.Linear(text_dim, self.fusion_dim)
        self.img_proj = nn.Linear(img_dim, self.fusion_dim)
        
        # Co-Attention层
        if fusion_type == 'coattn':
            self.text2img_attn = nn.MultiheadAttention(
                self.fusion_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.img2text_attn = nn.MultiheadAttention(
                self.fusion_dim, num_heads, dropout=dropout, batch_first=True
            )
        
        # 门控融合
        if fusion_type == 'gate' or fusion_type == 'coattn':
            self.gate_fc = nn.Sequential(
                nn.Linear(self.fusion_dim * 2, self.fusion_dim),
                nn.Sigmoid()
            )
        
        # 自适应融合
        if fusion_type == 'adaptive':
            self.adaptive_fc = nn.Sequential(
                nn.Linear(self.fusion_dim * 2, self.fusion_dim),
                nn.Tanh(),
                nn.Linear(self.fusion_dim, 3),  # 3种策略：文本主导、图像主导、平衡
                nn.Softmax(dim=-1)
            )
        
        # 输出层
        self.output_fc = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, text_features, img_features, text_mask=None):
        """
        前向传播
        
        参数:
            text_features: [batch_size, text_seq_len, text_dim]
            img_features: [batch_size, img_seq_len, img_dim] 或 [batch_size, img_dim]
            text_mask: [batch_size, text_seq_len] - 文本的padding mask
        
        返回:
            fused_features: [batch_size, text_seq_len, fusion_dim]
        """
        batch_size, text_seq_len, _ = text_features.shape
        
        # 处理图像特征维度
        if img_features.dim() == 2:
            # [batch_size, img_dim] -> [batch_size, 1, img_dim]
            img_features = img_features.unsqueeze(1)
        img_seq_len = img_features.size(1)
        
        # 投影到统一维度
        text_proj = self.text_proj(text_features)  # [B, text_len, fusion_dim]
        img_proj = self.img_proj(img_features)     # [B, img_len, fusion_dim]
        
        # 根据融合类型选择策略
        if self.fusion_type == 'coattn':
            fused = self._coattention_fusion(text_proj, img_proj, text_mask)
        elif self.fusion_type == 'gate':
            fused = self._gated_fusion(text_proj, img_proj, text_mask)
        elif self.fusion_type == 'adaptive':
            fused = self._adaptive_fusion(text_proj, img_proj, text_mask)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        
        # 输出层
        output = self.output_fc(fused)
        
        return output
    
    def _coattention_fusion(self, text_feat, img_feat, text_mask=None):
        """
        Co-Attention融合
        
        步骤：
        1. 文本查询图像：text → img
        2. 图像增强文本：img → text
        3. 门控融合
        """
        batch_size, text_len, feat_dim = text_feat.shape
        
        # 1. 文本查询图像 (Text as Query, Image as Key/Value)
        text_attn_out, _ = self.text2img_attn(
            query=text_feat,        # [B, text_len, dim]
            key=img_feat,           # [B, img_len, dim]
            value=img_feat,         # [B, img_len, dim]
            key_padding_mask=None   # 图像通常没有padding
        )
        # text_attn_out: [B, text_len, dim] - 图像对文本每个token的贡献
        
        # 2. 图像增强文本 (可选：图像也可以反向查询文本)
        # 这里简化，直接使用text_attn_out
        
        # 3. 门控融合
        # gate决定使用多少文本信息和多少图像信息
        gate_input = torch.cat([text_feat, text_attn_out], dim=-1)  # [B, text_len, 2*dim]
        gate = self.gate_fc(gate_input)  # [B, text_len, dim]
        
        # fused = gate * text + (1 - gate) * image_enhanced_text
        fused = gate * text_feat + (1 - gate) * text_attn_out
        
        return fused
    
    def _gated_fusion(self, text_feat, img_feat, text_mask=None):
        """
        简单门控融合（不使用注意力）
        """
        batch_size, text_len, feat_dim = text_feat.shape
        img_len = img_feat.size(1)
        
        # 将图像特征广播到每个文本token
        if img_len == 1:
            # 单个全局图像特征
            img_broadcast = img_feat.expand(-1, text_len, -1)
        else:
            # 多个patch特征，取平均
            img_global = img_feat.mean(dim=1, keepdim=True)  # [B, 1, dim]
            img_broadcast = img_global.expand(-1, text_len, -1)
        
        # 门控
        gate_input = torch.cat([text_feat, img_broadcast], dim=-1)
        gate = self.gate_fc(gate_input)
        
        fused = gate * text_feat + (1 - gate) * img_broadcast
        
        return fused
    
    def _adaptive_fusion(self, text_feat, img_feat, text_mask=None):
        """
        自适应融合：为每个token学习三种策略的权重
        
        三种策略:
        0: 文本主导 (text-dominant)
        1: 图像主导 (image-dominant)
        2: 平衡 (balanced)
        """
        batch_size, text_len, feat_dim = text_feat.shape
        img_len = img_feat.size(1)
        
        # 图像特征广播
        if img_len == 1:
            img_broadcast = img_feat.expand(-1, text_len, -1)
        else:
            img_global = img_feat.mean(dim=1, keepdim=True)
            img_broadcast = img_global.expand(-1, text_len, -1)
        
        # 自适应权重
        adaptive_input = torch.cat([text_feat, img_broadcast], dim=-1)
        weights = self.adaptive_fc(adaptive_input)  # [B, text_len, 3]
        
        # 三种融合策略
        text_dominant = text_feat
        img_dominant = img_broadcast
        balanced = (text_feat + img_broadcast) / 2
        
        # 加权组合
        fused = (
            weights[:, :, 0:1] * text_dominant +
            weights[:, :, 1:2] * img_dominant +
            weights[:, :, 2:3] * balanced
        )
        
        return fused


class CoAttentionMNER(nn.Module):
    """
    完整的Co-Attention MNER模块
    
    集成：
    1. 文本编码器（DeBERTa）
    2. 图像编码器（ViT）
    3. 跨模态注意力融合
    4. BiLSTM序列建模
    5. CRF输出层
    
    这是一个端到端的模块，可以直接替换现有的融合+head。
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        img_dim: int = 768,
        num_labels: int = 9,
        fusion_type: str = 'coattn',
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        use_crf: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 跨模态融合
        self.cross_modal_fusion = CrossModalAttentionFusion(
            text_dim=text_dim,
            img_dim=img_dim,
            fusion_dim=text_dim,
            num_heads=8,
            fusion_type=fusion_type,
            dropout=dropout
        )
        
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=text_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        
        # 分类层
        lstm_output_dim = hidden_size * 2
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_labels)
        
        # CRF
        self.use_crf = use_crf
        if use_crf:
            try:
                from torchcrf import CRF
                self.crf = CRF(num_labels, batch_first=True)
            except ImportError:
                from models.task_heads.token_label_heads import SimpleCRF
                self.crf = SimpleCRF(num_labels)
        
        print(f"✓ CoAttentionMNER initialized:")
        print(f"  - Fusion: {fusion_type}")
        print(f"  - BiLSTM: {num_lstm_layers} layers, hidden={hidden_size}")
        print(f"  - CRF: {use_crf}")
    
    def forward(self, text_features, img_features, attention_mask=None, labels=None):
        """
        前向传播
        
        参数:
            text_features: [batch_size, seq_len, text_dim] - DeBERTa输出
            img_features: [batch_size, img_seq_len, img_dim] - ViT输出
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] - 训练时的标签
        
        返回:
            训练时: (loss, logits)
            推理时: logits
        """
        # 1. 跨模态融合
        fused_features = self.cross_modal_fusion(text_features, img_features, attention_mask)
        
        # 2. BiLSTM
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed_features = nn.utils.rnn.pack_padded_sequence(
                fused_features, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.bilstm(packed_features)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=fused_features.size(1)
            )
        else:
            lstm_output, _ = self.bilstm(fused_features)
        
        lstm_output = self.dropout(lstm_output)
        
        # 3. 分类
        logits = self.classifier(lstm_output)
        
        # 4. Loss计算或返回
        if labels is not None:
            if self.use_crf:
                # CRF loss
                mask = attention_mask.bool() if attention_mask is not None else None
                crf_labels = labels.clone()
                crf_labels[crf_labels == -100] = 0
                
                log_likelihood = self.crf(logits, crf_labels, mask=mask, reduction='mean')
                loss = -log_likelihood
                return loss, logits
            else:
                # CE loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                active_loss = attention_mask.view(-1) == 1 if attention_mask is not None else None
                
                if active_loss is not None:
                    active_logits = logits.view(-1, logits.size(-1))[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                return loss, logits
        else:
            return logits


def test_cross_modal_attention():
    """测试跨模态注意力融合"""
    print("="*80)
    print("测试 CrossModalAttentionFusion")
    print("="*80)
    
    batch_size = 4
    text_len = 32
    img_len = 196  # ViT patch features (14x14)
    text_dim = 768
    img_dim = 768
    
    # 模拟输入
    text_feat = torch.randn(batch_size, text_len, text_dim)
    img_feat = torch.randn(batch_size, img_len, img_dim)
    text_mask = torch.ones(batch_size, text_len)
    text_mask[:, -5:] = 0  # 最后5个位置是padding
    
    print(f"\n输入:")
    print(f"  text_feat: {text_feat.shape}")
    print(f"  img_feat: {img_feat.shape}")
    print(f"  text_mask: {text_mask.shape}")
    
    # 测试三种融合类型
    for fusion_type in ['coattn', 'gate', 'adaptive']:
        print(f"\n{'='*80}")
        print(f"测试融合类型: {fusion_type}")
        print(f"{'='*80}")
        
        fusion = CrossModalAttentionFusion(
            text_dim=text_dim,
            img_dim=img_dim,
            fusion_dim=text_dim,
            num_heads=8,
            fusion_type=fusion_type,
            dropout=0.1
        )
        
        fused = fusion(text_feat, img_feat, text_mask)
        print(f"  ✓ 输出shape: {fused.shape}")
        print(f"  ✓ 参数量: {sum(p.numel() for p in fusion.parameters()):,}")
    
    # 测试完整模块
    print(f"\n{'='*80}")
    print("测试 CoAttentionMNER")
    print(f"{'='*80}")
    
    model = CoAttentionMNER(
        text_dim=text_dim,
        img_dim=img_dim,
        num_labels=9,
        fusion_type='coattn',
        hidden_size=256,
        num_lstm_layers=2,
        use_crf=True,
        dropout=0.3
    )
    
    labels = torch.randint(0, 9, (batch_size, text_len))
    labels[text_mask == 0] = -100
    
    print(f"\n训练模式:")
    loss, logits = model(text_feat, img_feat, text_mask, labels)
    print(f"  ✓ Loss: {loss.item():.4f}")
    print(f"  ✓ Logits shape: {logits.shape}")
    
    print(f"\n推理模式:")
    with torch.no_grad():
        logits = model(text_feat, img_feat, text_mask)
        print(f"  ✓ Logits shape: {logits.shape}")
    
    print(f"\n✅ 所有测试通过！")


if __name__ == "__main__":
    test_cross_modal_attention()







































