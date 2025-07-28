# continual/moe_adapters/moe_adapter_layer.py
import torch
import torch.nn as nn
from .expert import Expert, build_expert

class MoEAdapterLayer(nn.Module):
    """
    Token‑Level MoE Adapter with task‑specific router.
    """
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 1, expert_type: str = "lora", lora_rank: int = 8):
        super().__init__()
        self.experts = nn.ModuleList([build_expert(hidden_size, expert_type=expert_type, rank=lora_rank) for _ in range(num_experts)])
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
        # x: (B, L, H)
        cls = x[:, 0, :]                 # 取 [CLS] 的 hidden 向量
        gate_logits = self.router(cls)   # (B, E)
        if self.top_k < gate_logits.size(1):
            # 设定非 Top‑k 的位置为 −∞
            mask = torch.full_like(gate_logits, float('-inf'))
            topk_val, topk_idx = gate_logits.topk(self.top_k, dim=-1)
            mask.scatter_(1, topk_idx, topk_val)
            gate = torch.softmax(mask, dim=-1)
        else:
            gate = torch.softmax(gate_logits, dim=-1)
        gate = gate.view(B, -1, 1, 1)  # (B,E,1,1)
        weighted = (gate * experts_out).sum(dim=1)  # (B,L,H)

        return x + weighted  # 残差连接


