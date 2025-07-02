# continual/moe_adapters/expert.py
import torch.nn as nn
import torch

class Expert(nn.Module):
    """
    A single Adapter expert: Down‑proj → GELU → Up‑proj.
    Scale参数 γ 避免梯度爆炸，可学习也可固定.
    """
    def __init__(self, hidden_size: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        self.scale = nn.Parameter(torch.zeros(1))     # init to 0 → 几乎不扰动预训练主干

    def forward(self, x):
        # x: (B, L, H)
        return self.scale * self.up(self.act(self.down(x)))
