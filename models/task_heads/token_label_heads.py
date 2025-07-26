import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLabelHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, label_emb, task_name):
        super().__init__()
        self.num_labels = num_labels
        self.label_emb = label_emb  # GlobalLabelEmbedding 实例
        self.task_name = task_name

        # 将 token 向量投影到与 label 相同维度
        self.token_proj = nn.Linear(input_dim, hidden_dim)
        self.label_proj = nn.Linear(label_emb.emb_dim, hidden_dim)

        # 可选：归一化分母
        self.scale = hidden_dim ** 0.5

    def forward(self, seq_feats):
        # seq_feats: (batch, seq_len, input_dim)
        token_h = self.token_proj(seq_feats)  # (B, L, H)

        # 获取该任务所有 label 的 embedding 并做投影
        label_ids = torch.arange(self.num_labels, device=seq_feats.device)
        label_embs = self.label_emb(self.task_name, label_ids)   # (num_labels, emb_dim)
        label_h = self.label_proj(label_embs)                    # (num_labels, H)

        # 使用 dot product 得到 logits
        logits = torch.matmul(token_h, label_h.T) / self.scale   # (B, L, num_labels)

        return logits
