import torch
import torch.nn as nn

class LoRAExpert(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 8, alpha: float = 32):
        super().__init__()
        self.rank = rank
        # A 和 B 为低秩矩阵，用于对原有权重矩阵进行微调
        self.A = nn.Linear(hidden_size, rank, bias=False)
        self.B = nn.Linear(rank, hidden_size, bias=False)
        # scaling 参数，用于调节 LoRA 输出幅度
        self.scaling = alpha / rank
        # 初始化权重，遵循 LoRA 的初始化策略
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.scaling * self.B(self.A(x))
