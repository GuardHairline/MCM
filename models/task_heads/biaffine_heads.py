# models/task_heads/biaffine_heads.py

import torch
import torch.nn as nn
from continual.label_embedding import GlobalLabelEmbedding

class BiaffineSpanHead(nn.Module):
    """
    Yu et al. 2020 的双仿射跨度打分 。
    支持可选 label‑aware 三仿射：加入标签嵌入向量 z_l
    score = h_s^T U h_e + (h_s ⊕ h_e) W + b        # biaffine
    score_l = score + <h_s,  Wl, h_e> + z_l^T q    # triaffine ①
    """
    def __init__(self, input_dim, hidden_dim, num_labels,
                 use_triaffine=False, label_emb: GlobalLabelEmbedding = None,
                 task_name: str = None):
        super().__init__()
        self.use_triaffine = use_triaffine
        self.task_name = task_name
        self.num_labels = num_labels
        self.U = nn.Parameter(torch.zeros(hidden_dim, num_labels, hidden_dim))  # 双仿射矩阵
        self.W = nn.Linear(hidden_dim * 2, num_labels, bias=True)  # 用于拼接的权重矩阵
        self.start_proj = nn.Linear(input_dim, hidden_dim)  # token-level start projection
        self.end_proj   = nn.Linear(input_dim, hidden_dim)  # token-level end projection

        if use_triaffine:
            assert label_emb is not None
            self.label_emb = label_emb  # 标签嵌入
            # 创建一个从 hidden_dim 到 label_emb_dim 的投影矩阵
            self.q_proj = nn.Linear(hidden_dim, label_emb.emb_dim)  # 投影矩阵
            self.q = nn.Parameter(torch.zeros(label_emb.emb_dim))  # 标签嵌入的投影向量

    def forward(self, seq_feats):  # 输入是 (b, L, d)，表示序列特征
        Hs = self.start_proj(seq_feats)  # (b, L, h)
        He = self.end_proj(seq_feats)    # (b, L, h)

        # 双仿射操作 → O(L²)
        biaff = torch.einsum('bsh,hch,beh->bsec', Hs, self.U, He)  # (b, L, L, C)
        concat = torch.cat([Hs.unsqueeze(2).repeat(1,1,He.size(1),1),
                            He.unsqueeze(1).repeat(1,Hs.size(1),1,1)], dim=-1)  # 拼接（h_s ⊕ h_e）
        biaff = biaff + self.W(concat)  # (b, L, L, C)

        if self.use_triaffine:  # 如果使用三仿射
            # 获取当前任务的所有标签嵌入
            task_label_embeddings = self.label_emb.get_all_label_embeddings(self.task_name)  # (task_num_labels, emb_dim)
            # 确保标签嵌入数量与 num_labels 一致
            if task_label_embeddings.size(0) != self.num_labels:
                print(f"Warning: task_label_embeddings size ({task_label_embeddings.size(0)}) != num_labels ({self.num_labels})")
                # 如果数量不匹配，我们只使用前 num_labels 个嵌入
                task_label_embeddings = task_label_embeddings[:self.num_labels]
            
            # 将 Hs 投影到标签嵌入空间，然后与 q 计算点积
            Hs_proj = self.q_proj(Hs)  # (b, L, label_emb_dim)
            tri = torch.einsum('bsh,h->bs', Hs_proj, self.q)[:, :, None, None]  # (b, L, 1, 1)
            # 计算标签嵌入与 q 的点积
            label_scores = task_label_embeddings @ self.q  # (num_labels, 1)
            biaff = biaff + tri + label_scores.unsqueeze(0).unsqueeze(0)  # 广播到 (b, L, L, num_labels)

        return biaff  # 每个跨度的 C 维得分
