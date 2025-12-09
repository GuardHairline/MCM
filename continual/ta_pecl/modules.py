# continual/ta_pecl/modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRAExpert(nn.Module):
    """
    标准的 LoRA 模块，作为 MoE 中的一个专家。
    """
    def __init__(self, input_dim, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.lora_A = nn.Linear(input_dim, r, bias=False)
        self.lora_B = nn.Linear(r, input_dim, bias=False)
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化：A 为 Kaiming 分布，B 为 0
        # 这样初始状态下 LoRA 输出为 0，不会破坏预训练主干的特征
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # x: [batch, seq_len, dim]
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

class TaskAwareRouter(nn.Module):
    """
    任务感知路由器 (Task-Aware Router)
    核心机制：结合输入特征(Content) 和 任务嵌入(Task Intent) 来动态分配专家权重。
    """
    def __init__(self, input_dim, num_tasks, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 任务嵌入层：模型将自动学习这 4 个任务在向量空间中的关系
        self.task_embedding = nn.Embedding(num_tasks, input_dim)
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim), # 输入是 Content + Task_Emb
            nn.ReLU(),
            nn.Linear(input_dim, num_experts)
        )
        self.register_buffer('activation_counts', torch.zeros(num_experts))
        self.register_buffer('accumulated_weights', torch.zeros(num_experts))
        self.total_samples = 0 # 记录处理的总样本数
    def reset_stats(self):
        """重置统计数据"""
        self.activation_counts.zero_()
        self.accumulated_weights.zero_()
        self.total_samples = 0
    def forward(self, x, task_id):
        """
        x: [batch, seq_len, dim]
        task_id: [batch] or int or scalar tensor
        """
        batch_size = x.shape[0]
        
        # 1. 获取内容摘要 (Simple Pooling)
        if x.dim() == 3:
            state_pooled = x.mean(dim=1) # [batch, dim]
        else:
            state_pooled = x

        # 2. 获取任务意图
        # 处理 task_id 的各种可能格式 (int, tensor scalar, tensor vector)
        if isinstance(task_id, int):
            task_id = torch.full((batch_size,), task_id, device=x.device, dtype=torch.long)
        elif isinstance(task_id, torch.Tensor):
            if task_id.dim() == 0:
                task_id = task_id.expand(batch_size)
            elif task_id.dim() == 1 and task_id.size(0) == 1:
                task_id = task_id.expand(batch_size)
            # 否则假设已经是 [batch_size]
            
        task_id = task_id.to(x.device).long()
        
        task_emb = self.task_embedding(task_id) # [batch, dim]

        # 3. 融合决策
        router_input = torch.cat([state_pooled, task_emb], dim=-1)
        logits = self.gate(router_input) # [batch, num_experts]

        # 4. Top-K 选通 (含噪声增强探索)
        if self.training:
            noise = torch.randn_like(logits) * 0.05
            logits = logits + noise
            
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        
        # 5. 计算归一化权重
        weights = F.softmax(top_k_logits, dim=-1) # [batch, k]
        with torch.no_grad():
            self.total_samples += batch_size
            # 展平 indices 和 weights 以便 scatter_add
            flat_indices = indices.flatten() # [B * K]
            flat_weights = weights.flatten() # [B * K]
            
            # 统计激活次数 (每个索引位置 +1)
            ones = torch.ones_like(flat_indices, dtype=torch.float)
            self.activation_counts.scatter_add_(0, flat_indices, ones)
            
            # 统计权重总和
            self.accumulated_weights.scatter_add_(0, flat_indices, flat_weights)
        return indices, weights

class TA_PECL_Block(nn.Module):
    """
    TA-PECL 核心块：包含一个 Router 和 一组 Experts。
    将被插入到 Transformer 的每一层。
    """
    def __init__(self, hidden_size, num_tasks, expert_config, top_k=2):
        super().__init__()
        self.expert_names = list(expert_config.keys())
        self.num_experts = len(self.expert_names)
        self.top_k = top_k
        
        # 构建专家池
        self.experts = nn.ModuleDict({
            name: LoRAExpert(hidden_size, r=8) 
            for name in expert_config.keys()
        })
        
        # 构建路由器
        self.router = TaskAwareRouter(hidden_size, num_tasks, self.num_experts, top_k=top_k)

    def forward(self, hidden_states, task_id):
        """
        Args:
            hidden_states: [batch, seq_len, dim]
            task_id: int or Tensor, current task identifier
        """
        batch_size = hidden_states.shape[0]
        
        # 1. 路由决策 (传入 task_id)
        indices, weights = self.router(hidden_states, task_id)
        
        # 2. 专家计算 (Masked Dispatch)
        final_output = torch.zeros_like(hidden_states)
        
        for expert_idx, name in enumerate(self.expert_names):
            # 判断当前专家在当前 batch 的哪些样本中被选中
            mask = (indices == expert_idx) # [batch, top_k] boolean matrix
            
            # 如果该专家被至少一个样本选中
            if mask.any():
                # 计算专家输出 (前向传播)
                expert_out = self.experts[name](hidden_states)
                
                # 计算该专家对每个样本的贡献权重
                # mask.float() 只有在选中位置是 1
                # sum(dim=1) 将 top-k 维度压缩，得到每个样本对该专家的总权重
                sample_weight = (weights * mask.float()).sum(dim=1).view(batch_size, 1, 1)
                
                # 累加结果
                final_output += sample_weight * expert_out

        return final_output