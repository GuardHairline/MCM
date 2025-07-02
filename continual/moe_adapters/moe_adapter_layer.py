# continual/moe_adapters/moe_adapter_layer.py
import torch
import torch.nn as nn
from .expert import Expert

class MoEAdapterLayer(nn.Module):
    """
    Token‑Level MoE Adapter with task‑specific router.
    """
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.experts = nn.ModuleList([Expert(hidden_size) for _ in range(num_experts)])
        self.router  = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k   = top_k
        self.softmax = nn.Softmax(dim=-1)

    @torch.no_grad()
    def add_expert(self):
        """在新任务开始时调用，向 MoE 增添一个冻结的旧专家 + 一个**可训练**的新专家。"""
        for p in self.experts.parameters():
            p.requires_grad = False           # freeze old ones
        self.experts.append(Expert(self.experts[0].down.in_features))
        # 扩展 router 输出
        old_out = self.router.out_features
        weight = self.router.weight.data
        self.router = nn.Linear(weight.size(1), old_out + 1, bias=False)
        self.router.weight.data[:old_out] = weight                    # 拷贝旧权重
        nn.init.normal_(self.router.weight.data[old_out:], std=1e-4)  # 新列初始化

    def forward(self, x):
        """
        x: (B, L, H)
        return: (B, L, H)  残差 + 加权专家
        """
        B, L, H = x.shape

        # ---------- step‑1 计算所有专家输出 ----------
        # experts_out: (B, E, L, H)
        experts_out = torch.stack([expert(x) for expert in self.experts], dim=1)

        # ---------- step‑2 路由打分 ----------
        pooled = x.mean(dim=1)  # (B, H)
        gate = self.softmax(self.router(pooled))  # (B, E)

        # ---------- step‑3 只保留 top‑k ----------
        if self.top_k < gate.size(1):
            topk_val, topk_idx = gate.topk(self.top_k, dim=-1)  # (B,k)
            mask = torch.zeros_like(gate)  # (B,E)
            mask.scatter_(1, topk_idx, topk_val)  # 其余位置为 0
            gate = mask  # (B,E)

        # ---------- step‑4 加权聚合 ----------
        gate = gate.view(B, -1, 1, 1)  # (B,E,1,1)
        weighted = (gate * experts_out).sum(dim=1)  # (B,L,H)

        return x + weighted  # 残差连接


