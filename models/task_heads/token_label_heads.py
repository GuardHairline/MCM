import torch
import torch.nn as nn
from continual.label_embedding import GlobalLabelEmbedding

class TokenLabelHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, label_emb: GlobalLabelEmbedding, task_name: str):
        super().__init__()
        self.num_labels = num_labels
        self.label_emb = label_emb
        self.task_name = task_name
        self.token_proj = nn.Linear(input_dim, hidden_dim)
        self.label_proj = nn.Linear(label_emb.emb_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, 1)  # 输出单个分数

    def forward(self, seq_feats):
        # seq_feats: (batch, seq_len, input_dim)
        token_h = self.token_proj(seq_feats)  # (batch, seq_len, hidden_dim)
        label_embeddings = self.label_emb.get_all_label_embeddings(self.task_name)  # (num_labels, emb_dim)
        label_h = self.label_proj(label_embeddings)  # (num_labels, hidden_dim)

        # 组合token和label
        batch_size, seq_len, hidden_dim = token_h.size()
        num_labels = label_h.size(0)

        # 扩展token和label维度以便拼接
        token_h_exp = token_h.unsqueeze(2).expand(batch_size, seq_len, num_labels, hidden_dim)
        label_h_exp = label_h.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, num_labels, hidden_dim)
        concat = torch.cat([token_h_exp, label_h_exp], dim=-1)  # (batch, seq_len, num_labels, hidden_dim*2)

        logits = self.classifier(concat).squeeze(-1)  # (batch, seq_len, num_labels)
        return logits
