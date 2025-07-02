# continual/moe_adapters/ddas_router.py
"""
Distribution Discriminative Auto‑Selector (简化版)：
对每个已学 task 训练一个 1‑hidden‑layer AutoEncoder，推理时用重构误差
决定走 MoE (当前路由) 还是原始冻结主干；详见论文 §3.4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyAE(nn.Module):
    def __init__(self, in_dim: int, hid: int = 256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(hid, in_dim))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

class DDASRouter(nn.Module):
    def __init__(self, feature_dim: int, threshold: float = 0.02):
        super().__init__()
        self.ae_list = nn.ModuleList([TinyAE(feature_dim)])  # index 0 = OOD reference
        self.threshold = threshold

    def add_task(self):
        self.ae_list.append(TinyAE(self.ae_list[0].dec.out_features))

    def forward(self, feat):
        # feat: (B, D)
        errs = [F.mse_loss(ae(feat), feat, reduction='none').mean(dim=-1)   # (B,)
                for ae in self.ae_list]
        errs = torch.stack(errs, dim=1)            # (B, T+1)
        min_err, idx = errs.min(dim=1)             # (B,)

        branch_mask = min_err < self.threshold     # True → MoE; False → 原主干
        return branch_mask, idx
