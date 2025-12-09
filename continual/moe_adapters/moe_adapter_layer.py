# continual/moe_adapters/moe_adapter_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from .expert import build_expert
from .sparse_dispatcher import SparseDispatcher

class MoEAdapterLayer(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int,
                 top_k: int = 1, expert_type: str = "lora",
                 lora_rank: int = 8, noise_epsilon: float = 1e-2,
                 use_sparse: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_sparse = use_sparse
        self.noise_epsilon = noise_epsilon
        
        # 专家列表
        self.experts = nn.ModuleList(
            [build_expert(hidden_size, expert_type=expert_type,
                          rank=lora_rank) for _ in range(num_experts)]
        )
        
        # 使用 ModuleList 实现增量 Router (Task-Specific Routers)
        # 每个 Router 负责计算对应 Expert 的 logit (scalar)
        self.routers = nn.ModuleList(
            [nn.Linear(hidden_size, 1, bias=False) for _ in range(num_experts)]
        )
        
        # 噪声门控参数也需要独立
        self.noise_layers = nn.ModuleList(
            [nn.Linear(hidden_size, 1, bias=False) for _ in range(num_experts)]
        )
        
        # 初始化
        for router in self.routers:
            nn.init.normal_(router.weight, std=0.02)
        for noise in self.noise_layers:
            nn.init.zeros_(noise.weight)
            
        # 激活计数器（用于freeze_topk_experts）
        self.activation_counter = Counter()
        
        # Load balancing相关
        self.load_loss = None  # 存储当前batch的load loss

    def _cv_squared(self, x):
        """
        Coefficient of Variation (变异系数) 的平方
        用于load balancing loss，鼓励专家使用均匀分布
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)
    
    def _noisy_top_k_gating(self, x, train):
        """
        Noisy Top-K Gating（论文核心机制）
        """
        # 1. 计算每个 Expert 的 Logit
        # x: (B, H)
        # logits_list: [ (B, 1), (B, 1), ... ]
        logits_list = [router(x) for router in self.routers]
        clean_logits = torch.cat(logits_list, dim=1)  # (B, E)
        
        # 2. 训练时添加噪声
        if train:
            # 计算噪声标准差 - 修复：使用 noise_layers 列表
            noise_logits_list = [noise_layer(x) for noise_layer in self.noise_layers]
            raw_noise_stddev = torch.cat(noise_logits_list, dim=1) # (B, E)
            
            noise_stddev = F.softplus(raw_noise_stddev) + self.noise_epsilon
            
            # 添加高斯噪声
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
            
        # 3. Top-K选择
        if self.top_k < self.num_experts:
            # 只保留top-k个专家
            top_logits, top_indices = logits.topk(self.top_k, dim=1)
            
            # 创建mask：top-k以外的位置设为-inf
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, top_indices, top_logits)
            
            # Softmax（-inf位置会变成0）
            gates = F.softmax(mask, dim=1)  # (B, E)
        else:
            # 使用所有专家
            gates = F.softmax(logits, dim=1)
        
        # 4. 计算load（每个专家处理的样本数）
        load = (gates > 0).sum(0).float()  # (E,)
        
        return gates, load
    
    @torch.no_grad()
    def add_expert(self):
        """
        [Activate-Freeze Strategy]
        在新任务开始时调用：
        1. 冻结所有旧专家 (Experts)
        2. 冻结所有旧路由 (Routers)
        3. 添加一个新专家和对应的路由，保持可训练
        """        
        
        device = next(self.parameters()).device
        
        # 1. 冻结旧专家
        for p in self.experts.parameters():
            p.requires_grad = False
            
        # 2. 冻结旧 Router 和 Noise
        for p in self.routers.parameters():
            p.requires_grad = False
        for p in self.noise_layers.parameters():
            p.requires_grad = False
        
        # 3. 添加新专家
        # 获取 LoRA 配置 (兼容性处理)
        if hasattr(self.experts[0], 'down'):
            in_features = self.experts[0].down.in_features
            rank = self.experts[0].down.out_features 
        else: # LoraExpert implementation
            in_features = self.experts[0].A.in_features
            rank = self.experts[0].r
            
        new_expert = build_expert(
            in_features,
            expert_type="lora",
            rank=rank
        ).to(device)
        self.experts.append(new_expert)
        
        # 4. 添加新 Router 和 Noise Layer
        new_router = nn.Linear(self.hidden_size, 1, bias=False).to(device)
        new_noise = nn.Linear(self.hidden_size, 1, bias=False).to(device)
        
        # 初始化
        nn.init.normal_(new_router.weight, std=0.02)
        nn.init.zeros_(new_noise.weight)
        
        self.routers.append(new_router)
        self.noise_layers.append(new_noise)
        
        # 更新数量
        self.num_experts += 1

    def forward(self, x):
        """
        前向传播：MoE调制
        """
        # 支持2D和3D输入
        is_2d = (x.dim() == 2)
        
        if is_2d:
            # 2D输入: (B, H) → 转换为 (B, 1, H)
            x = x.unsqueeze(1)  # (B, 1, H)
        
        B, L, H = x.size()
        
        # 1. 使用[CLS] token计算gating
        cls = x[:, 0, :]  # (B, H)
        gates, load = self._noisy_top_k_gating(cls, self.training)  # (B, E), (E,)
        
        # 2. 计算Load Balancing Loss
        importance = gates.sum(0)  # (E,) 每个专家的总权重
        self.load_loss = self._cv_squared(importance) + self._cv_squared(load)
        
        # 3. 记录激活频次（用于freeze_topk_experts）
        with torch.no_grad():
            nonzero_gates = (gates > 0).float()  # (B, E)
            for expert_idx in range(self.num_experts):
                count = nonzero_gates[:, expert_idx].sum().item()
                if count > 0:
                    self.activation_counter[expert_idx] += int(count)
        
        # 4. 专家计算（密集计算以确保梯度正确传播）
        # (B, E, L, H)
        experts_out = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # (B, E, 1, 1)
        gates_expanded = gates.view(B, -1, 1, 1)
        
        # 加权求和 (B, L, H)
        weighted = (gates_expanded * experts_out).sum(dim=1)
        
        # 5. 残差连接
        output = x + weighted
        
        # 恢复原始shape
        if is_2d:
            output = output.squeeze(1)
        
        return output