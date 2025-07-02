import torch
import torch.nn as nn

class MoEAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, dropout_prob=0.1):
        super(MoEAdapter, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, input_dim)
        ) for _ in range(num_experts)])
        self.gating_network = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gate = torch.softmax(self.gating_network(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        output = torch.sum(expert_outputs * gate.unsqueeze(0).unsqueeze(-1), dim=0)
        return output
