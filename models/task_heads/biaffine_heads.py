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
            self.q = nn.Parameter(torch.zeros(label_emb.embedding.embedding_dim))  # 标签嵌入的投影

    def forward(self, seq_feats):  # 输入是 (b, L, d)，表示序列特征
        Hs = self.start_proj(seq_feats)  # (b, L, h)
        He = self.end_proj(seq_feats)    # (b, L, h)

        # 双仿射操作 → O(L²)
        biaff = torch.einsum('bsh,hch,beh->bsec', Hs, self.U, He)  # (b, L, L, C)
        concat = torch.cat([Hs.unsqueeze(2).repeat(1,1,He.size(1),1),
                            He.unsqueeze(1).repeat(1,Hs.size(1),1,1)], dim=-1)  # 拼接（h_s ⊕ h_e）
        biaff = biaff + self.W(concat)  # (b, L, L, C)

        if self.use_triaffine:  # 如果使用三仿射
            z = self.label_emb.embedding.weight  # (C, d)，标签的嵌入向量
            tri = torch.einsum('bsh,zh->bsz', Hs, self.q)[:, :, None, :]  # 计算标签嵌入的投影
            biaff = biaff + tri + z @ self.q  # 三仿射操作，得到最终得分 (b, L, L, C)

        return biaff  # 每个跨度的 C 维得分
