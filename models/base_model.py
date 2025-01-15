# models/base_model.py
import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models
from torchvision.models import ResNet50_Weights

class BaseMultimodalModel(nn.Module):
    def __init__(self, text_model_name="microsoft/deberta-v3-base",
                 image_model_name="resnet50",
                 hidden_dim=768,
                 multimodal_fusion="multi_head_attention",
                 num_heads=8):
        """
        :param text_model_name: 文本编码器预训练模型，如 'microsoft/deberta-v3-base'
        :param image_model_name: 图像编码器，如 'resnet50'
        :param hidden_dim: 用于投影或中间处理的维度 (通常与 text_hidden_size 相同)
        :param multimodal_fusion: 融合策略，支持 'concat' 或 'multi_head_attention' 等
        :param num_heads: MultiHeadAttention 的头数
        """
        super().__init__()

        # 文本编码器 (BERT)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        # 输出维度通常可由config.hidden_size获得
        self.text_hidden_size = self.text_encoder.config.hidden_size

        # 图像编码器 (ResNet)
        resnet = getattr(models, image_model_name)(weights=ResNet50_Weights.IMAGENET1K_V1)
        # 去掉分类层，拿到最后一层特征向量
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_hidden_size = 2048  # resnet50输出的向量dim

        # 可以在这里定义一个线性变换，使image特征和text特征映射到同一维度
        self.image_proj = nn.Linear(self.image_hidden_size, self.text_hidden_size)

        # 如果需要进一步融合，可再定义一个 cross-attention 或者简单 linear
        self.fusion_strategy = multimodal_fusion
        if self.fusion_strategy == "concat":
            self.fusion_output_dim = self.text_hidden_size + self.text_hidden_size
        elif self.fusion_strategy == "multi_head_attention":
            self.fusion_output_dim = self.text_hidden_size
            # 定义multi-head attention层
            self.mha = nn.MultiheadAttention(embed_dim=self.text_hidden_size,
                                             num_heads=num_heads,
                                             batch_first=False)
            # batch_first=False => 输入形状 [seq_len, batch_size, embed_dim]
        else:
            self.fusion_output_dim = self.text_hidden_size  # for demonstration

    def forward(self, input_ids, attention_mask, token_type_ids, image_tensor):
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
        # 一般取 [CLS] 向量, shape = [batch_size, hidden_size]
        text_emb = text_outputs.last_hidden_state[:, 0, :]

        # 2. 图像特征
        # image_tensor shape [batch_size, 3, H, W]
        img_feat = self.image_encoder(image_tensor)
        # img_feat shape [batch_size, 2048, 1, 1]
        img_feat = img_feat.view(img_feat.size(0), -1)  # -> [batch_size, 2048]
        img_feat = self.image_proj(img_feat)  # -> [batch_size, text_hidden_size]

        # 3. 多模态融合
        if self.fusion_strategy == "concat":
            # 简单拼接
            fused_feat = torch.cat([text_emb, img_feat], dim=-1)
        elif self.fusion_strategy == "multi_head_attention":
            # 1) 先把 text_emb/img_feat 扩展为 seq_len=1
            # MHA 要求形状 [seq_len, batch_size, embed_dim] => transpose
            text_seq = text_emb.unsqueeze(0)  # [1, batch_size, hidden_size]
            img_seq = img_feat.unsqueeze(0)   # [1, batch_size, hidden_size]

            # 2) 令 text_seq 做 query, img_seq 做 key/value
            #    => out_seq shape: [1, batch_size, hidden_size]
            out_seq, attn_weights = self.mha(
                query=text_seq,
                key=img_seq,
                value=img_seq
            )
            # out_seq => [1, batch_size, hidden_size]

            # 3) 再恢复到 [batch_size, hidden_size]
            fused_feat = out_seq.squeeze(0)
        else:
            # 其他策略，这里先演示一下最简单的加法
            fused_feat = text_emb + img_feat

        return fused_feat  # [batch_size, fusion_output_dim]
