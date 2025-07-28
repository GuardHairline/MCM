# continual/moe_adapters/lora_expert.py
import torch
import torch.nn as nn
import math                      

class LoRAExpert(nn.Module):
    """LoRA Adapter 专家：采用低秩矩阵 A/B 进行增量学习"""
    def __init__(self, hidden_size: int, rank: int = 8, alpha: float = 32):
        super().__init__()
        self.rank = rank
        # A 和 B 为低秩矩阵，用于对原始权重进行微调
        self.A = nn.Linear(hidden_size, rank, bias=False)
        self.B = nn.Linear(rank, hidden_size, bias=False)
        # scaling = α / r，控制 LoRA 输出幅度
        self.scaling = alpha / rank
        # 初始化：A 用 kaiming_uniform，B 全 0，确保初始扰动为 0
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        # x: (B, L, H)
        # 只计算低秩增量部分，主干权重在 MoEAdapterLayer 中处理
        return self.scaling * self.B(self.A(x))
