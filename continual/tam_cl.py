import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseMultimodalModel
from models.task_heads.get_head import get_head

import copy

class TaskAttentionBlock(nn.Module):
    """
    任务注意力块：在基础模型输出后，对任务 token 与序列做跨注意力
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=num_heads,
                                          batch_first=True,
                                          dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, seq):
        # seq: Tensor of shape (batch_size, seq_len+1, hidden_dim)
        x = self.norm1(seq)
        # query is the first token, key/value are all tokens
        q = x[:, :1, :]  # (B,1,H)
        kv = x          # (B,L+1,H)
        attn_out, _ = self.attn(q, kv, kv)
        # 将输出拼回序列位置0
        seq2 = seq.clone()
        seq2[:, :1, :] = attn_out
        # feed-forward + residual
        out = seq2 + self.ffn(self.norm2(seq2))
        return out

class TamCLModel(nn.Module):
    """
    TAM-CL 主模型：在 BaseMultimodalModel 输出后加入 TaskAttentionBlock + 任务 token 和可扩张头
    """
    def __init__(self,
                 text_model_name,
                 image_model_name,
                 fusion_strategy,
                 num_heads,
                 mode,
                 hidden_dim=None,
                 dropout_prob=0.1):
        super().__init__()
        # 共享骨干
        self.base_model = BaseMultimodalModel(
            text_model_name,
            image_model_name,
            multimodal_fusion=fusion_strategy,
            num_heads=num_heads,
            mode=mode
        )
        H = self.base_model.fusion_output_dim
        self.hidden_dim = H
        # 任务注意力块
        self.tab = TaskAttentionBlock(H, num_heads, dropout=dropout_prob)

        # 存放每个 session 的任务 token 和 head
        # key: session_id, value: token或head
        self.task_tokens = nn.ParameterDict()
        self.task_heads = nn.ModuleDict()

        # 教师模型，用于蒸馏
        self.teacher = None
        self.task_names = {}   # 新增字典用于记录 session_id 到 task_name 的映射
        self.sequence_tasks = {"mate", "mner", "mabsa"}


    def add_task(self, session_id: str, task_name: str, num_labels: int, args):
        """
        新增一个任务：创建可学习的 token 和 head
        session_id：唯一会话标识，用于存储和索引
        task_name：实际任务类型，用于构造 head
        """
        # 1. 任务 token
        token = torch.zeros(1, self.hidden_dim)
        nn.init.normal_(token, std=0.02)
        self.task_tokens[session_id] = nn.Parameter(token)

        # 2. 任务 head (根据任务类型初始化)
        head = get_head(task_name, self.base_model, args)
        self.task_heads[session_id] = head
        self.task_names[session_id] = task_name   # 记录任务名称

        # 冻结旧任务的 token 与 head，仅保留当前可训练
        for sid, param in self.task_tokens.items():
            param.requires_grad = (sid == session_id)
        for sid, module in self.task_heads.items():
            requires = (sid == session_id)
            for p in module.parameters(): p.requires_grad = requires

        # 3. 记录当前模型为新教师，并冻结其所有参数
        self.teacher = copy.deepcopy(self).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids, image_tensor, session_id: str):
        # 1) 共享骨干输出序列特征 (B, L, H)
        seq = self.base_model(input_ids, attention_mask, token_type_ids, image_tensor,
                               return_sequence=True)
        B, L, H = seq.shape
        # 2) 获取对应 session 的任务 token 并拼接 (B, L+1, H)
        token = self.task_tokens[session_id].expand(B, -1, -1)  # (B,1,H)
        seq_cat = torch.cat([token, seq], dim=1)
        # 3) 任务注意力块
        seq_tab = self.tab(seq_cat)  # (B, L+1, H)
        # 4) 取出任务 token 的输出表示 (B,H)
        task_feat = seq_tab[:, 0, :]
        # 5) 分支到任务 head，得到 logits
        # 若 head 是序列标注类型，则输入后续序列，否则输入 token 表示
        head = self.task_heads[session_id]
        task_name = self.task_names.get(session_id, None)

        if task_name in self.sequence_tasks:
            # 对于 MATE/MNER/MABSA 等 token‑级任务
            logits = head(seq_tab[:, 1:, :])   # (B, L, H) → (B, L, num_labels)
        else:
            # 对于 MASC 等句级任务
            logits = head(task_feat)           # (B, H) → (B, num_labels)
        return logits, seq, seq_tab

    def compute_distillation(self, seq, session_id: str, T: float):
        """
        基于教师模型对共享表示 seq 进行中间层蒸馏
        seq: (B, L, H) 基础骨干输出
        """
        with torch.no_grad():
            old_seq = self.teacher.base_model(
                self.last_inputs['input_ids'],
                self.last_inputs['attention_mask'],
                self.last_inputs.get('token_type_ids', None),
                self.last_inputs['image_tensor'],
                return_sequence=True
            )
        B, L, H = seq.shape
        seq_flat = seq.view(-1, H)
        old_flat = old_seq.view(-1, H)
        log_q = F.log_softmax(seq_flat / T, dim=-1)
        p = F.softmax(old_flat / T, dim=-1)
        return F.kl_div(log_q, p, reduction='batchmean') * (T**2)

    def diversity_loss(self):
        """
        使不同任务 token 表示互异的多样性损失
        """
        tokens = torch.stack([t.squeeze(0) for t in self.task_tokens.values()])  # (n_sessions, H)
        sim = F.cosine_similarity(tokens.unsqueeze(1), tokens.unsqueeze(0), dim=-1)
        n = sim.size(0)
        diff = 1 - sim
        mask = (~torch.eye(n, dtype=torch.bool, device=sim.device))
        return diff[mask].mean()