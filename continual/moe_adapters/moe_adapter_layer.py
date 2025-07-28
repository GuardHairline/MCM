# continual/moe_adapters/moe_adapter_layer.py

import torch
import torch.nn as nn
from collections import Counter
from .expert import build_expert

class MoEAdapterLayer(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int,
                 top_k: int = 1, expert_type: str = "lora",
                 lora_rank: int = 8):
        super().__init__()
        self.experts = nn.ModuleList(
            [build_expert(hidden_size, expert_type=expert_type,
                          rank=lora_rank) for _ in range(num_experts)]
        )
        self.router  = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k   = top_k
        self.softmax = nn.Softmax(dim=-1)
        # 初始化激活计数器
        self.activation_counter = Counter()

    @torch.no_grad()
    def add_expert(self):
        """在新任务开始时调用，冻结旧专家并添加一个新专家"""
        for p in self.experts.parameters():
            p.requires_grad = False
        # 默认新专家使用 build_expert 创建
        self.experts.append(build_expert(
            self.experts[0].down.in_features if hasattr(self.experts[0], 'down')
            else self.experts[0].A.in_features,  # 兼容 LoRAExpert
            expert_type="lora",
            rank=getattr(self.experts[0], 'rank', 8)
        ))
        # 扩展路由器输出维度
        old_out = self.router.out_features
        old_weight = self.router.weight.data.clone()
        self.router = nn.Linear(self.router.in_features,
                                old_out + 1, bias=False)
        self.router.weight.data[:old_out] = old_weight
        nn.init.normal_(self.router.weight.data[old_out:], std=1e-4)

    def forward(self, x):
        """
        x: (B, L, H) → 返回 MoE 调制后的特征 (B, L, H)
        单模态/多模态、句级/序列级任务均适用
        """
        B, L, H = x.size()
        cls = x[:, 0, :]                          # [CLS] token
        gate_logits = self.router(cls)            # (B, E)
        if self.top_k < self.router.out_features:
            mask = torch.full_like(gate_logits, float('-inf'))
            topk_val, topk_idx = gate_logits.topk(self.top_k, dim=-1)
            mask.scatter_(1, topk_idx, topk_val)
            gate = torch.softmax(mask, dim=-1)    # (B, E)
        else:
            gate = torch.softmax(gate_logits, dim=-1)

        # 记录激活频次：统计当前 batch Top‑k 专家
        with torch.no_grad():
            _, selected = gate_logits.topk(self.top_k, dim=-1)
            for row in selected:
                for idx in row.tolist():
                    self.activation_counter[idx] += 1

        # 计算每个专家输出
        experts_out = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B,E,L,H)
        # 加权求和
        gate = gate.view(B, -1, 1, 1)             # (B,E,1,1)
        weighted = (gate * experts_out).sum(dim=1)  # (B,L,H)

        return x + weighted                       # 残差连接
