# models/hierarchical_multitask_model.py
import torch
import torch.nn as nn
from models.base_model import BaseMultimodalModel
from models.task_heads.token_label_heads import TokenLabelHead
from models.task_heads.sent_label_attn import LabelAttentionSentHead

class HierarchicalMultitaskModel(nn.Module):
    def __init__(self, text_model_name, image_model_name, num_token_labels, num_sentence_labels,
                 fusion='concat', hidden_dim=768, label_emb=None, task_name=None):
        super().__init__()
        self.base_model = BaseMultimodalModel(text_model_name, image_model_name,
                                              multimodal_fusion=fusion)
        self.hidden_dim = hidden_dim
        self.task_name = task_name
        
        # 获取text_hidden_size，如果获取失败则使用默认值
        try:
            text_hidden_size = self.base_model.text_hidden_size
            print(f"DEBUG: text_hidden_size = {text_hidden_size}")
            if text_hidden_size <= 0:
                print(f"DEBUG: text_hidden_size <= 0, using default 768")
                text_hidden_size = 768  # 默认值
        except Exception as e:
            print(f"DEBUG: Exception getting text_hidden_size: {e}, using default 768")
            text_hidden_size = 768  # 默认值
        
        print(f"DEBUG: Final text_hidden_size = {text_hidden_size}")
        
        # token 级解码器
        if label_emb is not None:
            self.token_head = TokenLabelHead(text_hidden_size,
                                             hidden_dim, num_token_labels, label_emb, task_name)
        else:
            # 如果没有label_emb，使用简单的线性层
            self.token_head = nn.Linear(text_hidden_size, num_token_labels)
        
        # 句级解码器
        if label_emb is not None:
            self.sent_head = LabelAttentionSentHead(text_hidden_size,
                                                        num_sentence_labels, label_emb, task_name)
        else:
            # 如果没有label_emb，使用简单的线性层
            self.sent_head = nn.Linear(text_hidden_size, num_sentence_labels)
        
        # 可学习的池化权重，融合 token 输出和 CLS 的信息
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, attention_mask, token_type_ids, image_tensor):
        seq_outputs = self.base_model(input_ids, attention_mask,
                                      token_type_ids, image_tensor, return_sequence=True)
        # token 预测
        if isinstance(self.token_head, TokenLabelHead):
            token_logits = self.token_head(seq_outputs)    # (B, L, C_token)
        else:
            # 简单的线性层
            token_logits = self.token_head(seq_outputs)    # (B, L, C_token)
        
        # 池化 token 特征，作为句子表示
        pooled_seq = (seq_outputs.mean(dim=1) * self.alpha +
                      seq_outputs[:,0,:] * (1 - self.alpha))
        # 句级预测
        if isinstance(self.sent_head, LabelAttentionSentHead):
            sent_logits = self.sent_head(pooled_seq)        # (B, C_sentence)
        else:
            # 简单的线性层
            sent_logits = self.sent_head(pooled_seq)        # (B, C_sentence)
        return token_logits, sent_logits
