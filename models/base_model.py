# models/base_model.py
import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models
from torchvision.models import ResNet50_Weights

class BaseMultimodalModel(nn.Module):
    def __init__(self, text_model_name="microsoft/deberta-v3-base",
                 image_model_name="google/vit-base-patch16-224-in21k",
                 hidden_dim=768,
                 multimodal_fusion="concat",
                 num_heads=8,
                 mode="multimodal",
                 dropout_prob=0.1,
                 image_weight=0.1):
        """
        :param text_model_name: 文本编码器预训练模型，如 'microsoft/deberta-v3-base'
        :param image_model_name: 图像编码器，如 'resnet50'
        :param hidden_dim: 用于投影或中间处理的维度 (通常与 text_hidden_size 相同)
        :param multimodal_fusion: 融合策略，支持 'concat' 或 'multi_head_attention' 等
        :param num_heads: MultiHeadAttention 的头数
        :param mode: "text_only" 或 "multimodal"，指定是否使用图像数据
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

        # 图像编码器 (ViT)
        self.image_encoder = AutoModel.from_pretrained(image_model_path)
        self.image_hidden_size = self.image_encoder.config.hidden_size

        # 可以在这里定义一个线性变换，使image特征和text特征映射到同一维度
        self.image_proj = nn.Linear(self.image_hidden_size, self.text_hidden_size)

        # 如果需要进一步融合，可再定义一个 cross-attention 或者简单 linear
        self.fusion_strategy = multimodal_fusion
        self.mode = mode

        if self.fusion_strategy == "multi_head_attention":
            self.fusion_output_dim = self.text_hidden_size
            # 定义multi-head attention层
            self.mha = nn.MultiheadAttention(embed_dim=self.text_hidden_size,
                                             num_heads=num_heads,
                                             batch_first=False)
            # batch_first=False => 输入形状 [seq_len, batch_size, embed_dim]
        elif self.fusion_strategy == "concat":
            self.fusion_output_dim = self.text_hidden_size
        elif self.fusion_strategy == "add":
            self.fusion_output_dim = self.text_hidden_size
        else:
            self.fusion_output_dim = self.text_hidden_size  # fallback

        self.dropout = nn.Dropout(dropout_prob)
        self.fc_concat = nn.Linear(self.text_hidden_size * 2, self.text_hidden_size)  # Full connection for concatenated features
        self.image_weight = image_weight  # 设置图像模态的权重
    def get_image_features(self, image_tensor):
        # ViT特征提取（使用CLS token）
        outputs = self.image_encoder(image_tensor)
        features = outputs.last_hidden_state[:, 0, :]
        return self.image_proj(features)
    def forward(self, input_ids, attention_mask, token_type_ids, image_tensor, return_sequence=False):
        """
        :param return_sequence: 为 True 时，返回序列特征 (batch_size, seq_len, fusion_dim)
                               为 False 时，只返回句向量 (batch_size, fusion_dim) (CLS向量)
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
        # 序列隐状态: [batch_size, seq_len, hidden_size]
        text_sequence = text_outputs.last_hidden_state
        # [CLS] 向量
        text_cls = text_sequence[:, 0, :]  # (batch_size, hidden_size)

        if self.mode == "text_only":
            # 只使用文本模态，返回文本的[CLS]向量
            if return_sequence:
                return text_sequence  # 返回整个文本序列
            else:
                return text_cls  # 返回[CLS]向量

        # ====== 图像特征 ======
        img_feat = self.get_image_features(image_tensor)

        # ====== 多模态融合 ======

        if self.fusion_strategy == "multi_head_attention":
            # 注意：目前的 MHA 代码只展示“句向量 + 图像向量”的示例
            # 若要对序列中每个 token 都做 cross-attention，需要更改 key/value 的序列长度
            if return_sequence:
                # 这里简单示例：对 text_sequence 的所有 token 做“图像向量为 key/value”的跨注意力
                # text_seq => [seq_len, batch_size, hidden_size]
                text_seq = text_sequence.transpose(0, 1)
                # img_seq  => [1, batch_size, hidden_size]
                img_seq = img_feat.unsqueeze(0)

                img_seq = img_seq * self.image_weight

                out_seq, _ = self.mha(query=text_seq, key=img_seq, value=img_seq)
                # [seq_len, batch_size, hidden_size] => 转回 (b, seq_len, hidden_size)
                out_seq = out_seq.transpose(0, 1)
                return out_seq
            else:
                # 只处理 CLS
                text_seq = text_cls.unsqueeze(0)  # [1, batch_size, hidden_size]
                img_seq = img_feat.unsqueeze(0)  # [1, batch_size, hidden_size]

                img_seq = img_seq * self.image_weight

                out_seq, _ = self.mha(query=text_seq, key=img_seq, value=img_seq)
                fused_cls = out_seq.squeeze(0)  # (batch_size, hidden_size)
                return fused_cls

        elif self.fusion_strategy == "concat":
            if return_sequence:
                # 将图像特征 broadcast 后拼接到每个 token
                # img_feat.unsqueeze(1) => [batch_size, 1, hidden_size]
                # repeat 在 seq_len 维度上扩展
                expanded_img = img_feat.unsqueeze(1).repeat(1, text_sequence.size(1), 1)
                fused_seq = torch.cat([text_sequence, expanded_img], dim=-1)  # (b, seq_len, 2*hidden_size)
                fused_seq = self.fc_concat(fused_seq)  # (b, seq_len, hidden_size)
                return fused_seq
            else:
                # 只返回 CLS
                fused_cls = torch.cat([text_cls, img_feat], dim=-1)  # (b, hidden_size*2)
                fused_cls = self.fc_concat(fused_cls)  # (b, hidden_size)
                return fused_cls

        elif self.fusion_strategy == "add":
            if return_sequence:
                expanded_img = img_feat.unsqueeze(1).expand(-1, text_sequence.size(1), -1)
                fused_seq = text_sequence + expanded_img  # (b, seq_len, hidden_size)
                return fused_seq
            else:
                fused_cls = text_cls + img_feat
                return fused_cls
