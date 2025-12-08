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
        
        # Router网络（用于计算gate logits）
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # ✓ 新增：Noise网络（用于Noisy Top-K Gating）
        self.w_noise = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.zeros_(self.w_noise.weight)  # 初始化为0，避免初期扰动过大
        
        # 激活计数器（用于freeze_topk_experts）
        self.activation_counter = Counter()
        
        # Load balancing相关
        self.load_loss = None  # 存储当前batch的load loss

        # 记录当前正在训练的专家索引（通常是最后一个）
        # 如果是 -1，表示所有专家都已冻结或处于推理模式
        self.active_expert_index = num_experts - 1
    def _cv_squared(self, x):
        """
        Coefficient of Variation (变异系数) 的平方
        用于load balancing loss，鼓励专家使用均匀分布
        
        Args:
            x: (num_experts,) 每个专家的负载或重要性
        
        Returns:
            cv^2: scalar，值越小表示分布越均匀
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)
    
    def _noisy_top_k_gating(self, x, train):
        """
        Noisy Top-K Gating（论文核心机制）
        
        训练时：添加噪声探索不同专家
        推理时：使用干净的logits
        
        Args:
            x: (B, H) [CLS] token特征
            train: bool，是否训练模式
        
        Returns:
            gates: (B, E) softmax后的gate权重，top-k以外的为0
            load: (E,) 每个专家处理的样本数（用于load balancing）
        """
        # 1. 计算clean logits
        clean_logits = self.router(x)  # (B, E)
        
        # 2. 训练时添加噪声
        if train:
            # 计算噪声标准差
            raw_noise_stddev = self.w_noise(x)  # (B, E)
            noise_stddev = F.softplus(raw_noise_stddev) + self.noise_epsilon
            # 添加高斯噪声
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # 训练时的路由约束
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
        """在新任务开始时调用，冻结旧专家并添加一个新专家"""
        # 1. 冻结所有旧专家
        for p in self.experts.parameters():
            p.requires_grad = False
        
        # 2. 获取设备信息（从第一个专家获取）
        device = next(self.experts[0].parameters()).device
        
        # 3. 添加新专家并移到正确设备
        new_expert = build_expert(
            self.experts[0].down.in_features if hasattr(self.experts[0], 'down')
            else self.experts[0].A.in_features,  # 兼容 LoRAExpert
            expert_type="lora",
            rank=getattr(self.experts[0], 'rank', 8)
        ).to(device)
        self.experts.append(new_expert)
        
        # 4. 扩展Router输出维度
        old_out = self.router.out_features
        old_router_weight = self.router.weight.data.clone()
        old_noise_weight = self.w_noise.weight.data.clone()
        
        # 重新创建router和noise网络
        self.router = nn.Linear(self.router.in_features, old_out + 1, bias=False).to(device)
        self.w_noise = nn.Linear(self.w_noise.in_features, old_out + 1, bias=False).to(device)
        
        # 复制旧权重
        self.router.weight.data[:old_out] = old_router_weight
        self.w_noise.weight.data[:old_out] = old_noise_weight
        
        # 初始化新专家的权重（小随机值）
        nn.init.normal_(self.router.weight.data[old_out:], std=1e-4)
        nn.init.zeros_(self.w_noise.weight.data[old_out:])
        
        # 更新num_experts
        self.num_experts = old_out + 1
        self.active_expert_index = self.num_experts - 1

    def forward(self, x):
        """
        前向传播：MoE调制
        
        Args:
            x: (B, L, H) 或 (B, H) 输入特征
        
        Returns:
            output: same shape as x
            
        Side effects:
            self.load_loss: 设置当前batch的load balancing loss
        """
        # ✅ 修复: 支持2D和3D输入
        original_shape = x.shape
        is_2d = (x.dim() == 2)
        
        if is_2d:
            # 2D输入: (B, H) → 转换为 (B, 1, H)
            x = x.unsqueeze(1)  # (B, 1, H)
        
        B, L, H = x.size()  # ✅ 现在总是3D
        
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
        
        # 4. 专家计算（使用SparseDispatcher或密集计算）
        if self.use_sparse and self.training:
            # ✓ 稀疏计算：只让gate>0的专家处理对应样本
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)  # list of (expert_batch_size, L, H)
            
            # 处理每个专家（跳过空输入）
            expert_outputs = []
            for i, expert_input in enumerate(expert_inputs):
                if expert_input.size(0) > 0:  # 如果有样本分配给该专家
                    expert_outputs.append(self.experts[i](expert_input))
                else:
                    # 创建空tensor保持列表长度一致
                    expert_outputs.append(torch.empty(0, L, H, device=x.device, dtype=x.dtype))
            
            # 合并专家输出
            weighted = dispatcher.combine(expert_outputs, multiply_by_gates=True)  # (B, L, H)
        else:
            # ✓ 密集计算：所有专家处理所有样本（推理时或use_sparse=False）
            experts_out = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, E, L, H)
            gates_expanded = gates.view(B, -1, 1, 1)  # (B, E, 1, 1)
            weighted = (gates_expanded * experts_out).sum(dim=1)  # (B, L, H)
        
        # 5. 残差连接
        output = x + weighted
        
        # ✅ 修复: 恢复原始shape
        if is_2d:
            output = output.squeeze(1)  # (B, 1, H) → (B, H)
        
        return output
