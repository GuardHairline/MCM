# continual/label_graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelGraph(nn.Module):
    """
    静态标签图 G (N×N)，GN[i,j] 越大表示两个标签相似。
    对 logits 进行 y' = softmax( (1-τ)·logits + τ·G·softmax(logits) )
    同时计算一致性损失。
    """
    def __init__(self, label_emb, tau=0.5):
        super().__init__()
        self.tau = tau
        with torch.no_grad():
            z = label_emb.embedding.weight
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)  # (N,N)
            sim.fill_diagonal_(0)      # 去自环
            self.register_buffer('G', sim)

    def forward(self, task_name, logits):
        prob = logits.softmax(-1)                       # (… , N)
        smoothed = (1-self.tau) * prob + self.tau * (prob @ self.G)
        self.consistency_loss = F.kl_div(
            smoothed.log(), prob.detach(), reduction='batchmean')
        new_logits = smoothed.log()                     # 重新当作 logit
        return new_logits
