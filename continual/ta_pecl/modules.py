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

class SoftPriorRouter(nn.Module):
    """
    软先验路由器：
    Score = Dynamic_Gate(x) + Task_Bias[t] + Mode_Bias[m]
    """
    def __init__(self, input_dim, num_tasks, num_modes, expert_config, top_k=4):
        super().__init__()
        self.expert_names = list(expert_config.keys())
        self.num_experts = len(self.expert_names)
        self.top_k = top_k
        
        # 1. 动态门控 (Content-based)
        # 初始权重设小一点，让先验起主导作用
        self.dynamic_gate = nn.Linear(input_dim, self.num_experts, bias=False)
        nn.init.normal_(self.dynamic_gate.weight, std=0.01)

        # 2. 任务偏置矩阵 [Num_Tasks, Num_Experts]
        self.task_bias = nn.Parameter(torch.zeros(num_tasks, self.num_experts))
        
        # 3. 模态偏置矩阵 [Num_Modes, Num_Experts]
        self.mode_bias = nn.Parameter(torch.zeros(num_modes, self.num_experts))
        
        # --- 初始化偏置 (注入你的设计思想) ---
        self._initialize_biases(expert_config)

        # 统计缓冲
        self.register_buffer('activation_counts', torch.zeros(self.num_experts))
        self.register_buffer('accumulated_weights', torch.zeros(self.num_experts))
        self.total_samples = 0

    def _initialize_biases(self, expert_config):
        """
        根据 config 中的 init_task_id / init_mode_id 设置强先验
        """
        # 默认偏置为负值 (抑制)，选中的设为正值 (激活)
        with torch.no_grad():
            self.task_bias.fill_(-1.0)
            self.mode_bias.fill_(-1.0)
            
            for idx, name in enumerate(self.expert_names):
                cfg = expert_config[name]
                
                # 设置任务偏置
                if 'init_task_id' in cfg:
                    tid = cfg['init_task_id']
                    # 强激活：比如 +3.0，足以在 Softmax 中占据优势
                    self.task_bias[tid, idx] = 3.0 
                
                # 设置模态偏置
                if 'init_mode_id' in cfg:
                    mid = cfg['init_mode_id']
                    self.mode_bias[mid, idx] = 2.0 # 模态偏置稍微弱一点，允许任务覆盖

    def reset_stats(self):
        self.activation_counts.zero_()
        self.accumulated_weights.zero_()
        self.total_samples = 0

    def forward(self, x, task_id, mode_id):
        """
        x: [batch, seq, dim]
        task_id: int or [batch]
        mode_id: int or [batch]
        """
        batch_size = x.shape[0]
        
        # Pooling
        if x.dim() == 3:
            x_pooled = x.mean(dim=1)
        else:
            x_pooled = x
            
        # 1. 计算动态分数
        dynamic_logits = self.dynamic_gate(x_pooled) # [B, E]
        
        # 2. 获取偏置
        # 扩展 task_id / mode_id
        if isinstance(task_id, int):
            task_id = torch.full((batch_size,), task_id, device=x.device, dtype=torch.long)
        if isinstance(mode_id, int):
            mode_id = torch.full((batch_size,), mode_id, device=x.device, dtype=torch.long)
            
        t_bias = self.task_bias[task_id] # [B, E]
        m_bias = self.mode_bias[mode_id] # [B, E]
        
        # 3. 叠加总分
        total_logits = dynamic_logits + t_bias + m_bias
        
        # 训练噪声
        if self.training:
            total_logits = total_logits + torch.randn_like(total_logits) * 0.1
            
        # 4. Top-K 竞争
        # 此时所有专家（定死的、灵活的）一起根据 total_logits 竞争前 K 个名额
        top_k_logits, indices = total_logits.topk(self.top_k, dim=-1)
        weights = F.softmax(top_k_logits, dim=-1)
        
        # 5. 统计
        with torch.no_grad():
            self.total_samples += batch_size
            flat_indices = indices.flatten()
            flat_weights = weights.flatten()
            ones = torch.ones_like(flat_indices, dtype=torch.float)
            self.activation_counts.scatter_add_(0, flat_indices, ones)
            self.accumulated_weights.scatter_add_(0, flat_indices, flat_weights)
            
        return indices, weights

class TA_PECL_Block(nn.Module):
    """
    TA-PECL 核心块：包含一个 Router 和 一组 Experts。
    将被插入到 Transformer 的每一层。
    """
    def __init__(self, hidden_size, num_tasks, num_modes, expert_config, top_k=4):
        super().__init__()
        self.expert_names = list(expert_config.keys())
        self.top_k = top_k
        
        # 构建专家池
        self.experts = nn.ModuleDict({
            name: LoRAExpert(hidden_size, r=expert_config[name].get('r', 8)) 
            for name in expert_config.keys()
        })
        
        # 构建路由器
        self.router = SoftPriorRouter(hidden_size, num_tasks, num_modes, expert_config, top_k=top_k)
    def forward(self, hidden_states, task_id, mode_id):
        """
        Args:
            hidden_states: [batch, seq_len, dim]
            task_id: int or Tensor, current task identifier
        """
        batch_size = hidden_states.shape[0]
        
        # 1. 路由决策 (传入 task_id)
        indices, weights = self.router(hidden_states, task_id, mode_id)
        
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