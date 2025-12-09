import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseMultimodalModel
from models.task_head_manager import TaskHeadManager
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
        # 自动推断维度
        H = self.base_model.fusion_output_dim
        self.hidden_dim = H
        # 任务注意力块
        self.tab = TaskAttentionBlock(H, num_heads, dropout=dropout_prob)

        # 使用 HeadManager 管理分类头
        self.head_manager = TaskHeadManager(self.base_model)

        # 存放每个 session 的任务 token 和 head
        # key: session_id, value: token或head
        self.task_tokens = nn.ParameterDict()

        # 教师模型，用于蒸馏
        self.teacher = None
        self.active_session_id = None # 用于记录当前活动的 session

        # 记录上一次的输入，用于蒸馏计算
        self.last_inputs = {}

    def add_task_head(self, session_name, task_name, head, args=None):
        """
        [新增] 适配框架接口的方法。
        当框架调用此方法注册头时，我们同时初始化 TAM-CL 所需的任务 Token。
        """
        # 1. 注册头 (HeadManager 会处理 get_head 的逻辑，这里传入的 head 可能是 None，由 manager 创建)
        # 注意：train_refactored 传入的是 full_model.head，这通常是 None 或旧头。
        # 我们应该利用 HeadManager 的 create_and_register_head 逻辑，或者在这里手动创建。
        
        # 框架中 train_refactored 是这样调用的：
        # full_model.add_task_head(args.session_name, args.task_name, full_model.head, args)
        
        # [Fix 1] Check if head is None (from train_utils). If so, create it.
        if head is None:
            use_label_embedding = getattr(args, 'use_label_embedding', False)
            
            # Check if TaskHeadManager supports head_key via kwargs (based on your latest code)
            # If not, it will ignore it or we catch TypeError.
            # Ideally, we pass it if it's in args.
            try:
                head_key = getattr(args, 'head_key', session_name)
                # Attempt to pass head_key if the manager supports it (Advanced check)
                import inspect
                sig = inspect.signature(self.head_manager.create_and_register_head)
                if 'head_key' in sig.parameters or 'kwargs' in sig.parameters:
                     self.head_manager.create_and_register_head(session_name, task_name, args, use_label_embedding, head_key=head_key)
                else:
                     self.head_manager.create_and_register_head(session_name, task_name, args, use_label_embedding)
            except Exception as e:
                print(f"[TAM-CL Warning] Head creation fallback: {e}")
                self.head_manager.create_and_register_head(session_name, task_name, args, use_label_embedding)
        else:
            # [Fix 2] Correct Argument Order: (session, task, head, args)
            self.head_manager.register_head(session_name, task_name, head, args)
        
        # 2. 初始化该任务的 Token (TAM-CL 特有逻辑)
        if session_name not in self.task_tokens:
            token = torch.zeros(1, self.hidden_dim)
            nn.init.normal_(token, std=0.02)
            self.task_tokens[session_name] = nn.Parameter(token)
            print(f"[TAM-CL] Initialized task token for session: {session_name}")

        # 3. 冻结旧任务的 token，仅保留当前可训练
        for sid, param in self.task_tokens.items():
            param.requires_grad = (sid == session_name)
            
        # 4. 更新教师模型 (如果是新任务)
        # (这里简单处理：每次添加任务前，如果是持续学习阶段，应该在外部控制教师更新，
        # 但为了复现原逻辑，我们在 set_active_head 或这里做均可)
        if len(self.task_tokens) > 1 and self.teacher is None:
             self._update_teacher()
    # Alias for add_task to support legacy calls from train_utils.py
    def add_task(self, session_name, task_name, num_labels=None, args=None):
        """Legacy alias for add_task_head"""
        return self.add_task_head(session_name, task_name, None, args)
    def set_active_head(self, session_name, strict=True):
        """
        [新增] 设置当前活动的任务 (用于 forward 默认行为)
        """
        self.head_manager.set_active_head(session_name)
        self.active_session_id = session_name
        
        # 确保对应的 Token 存在
        if session_name not in self.task_tokens and strict:
            raise ValueError(f"Task token for {session_name} not found in TAM-CL model!")

    def _update_teacher(self):
        """更新教师模型为当前模型的副本"""
        self.teacher = copy.deepcopy(self).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        print("[TAM-CL] Teacher model updated.")

    def forward(self, input_ids, attention_mask, token_type_ids=None, image_tensor=None, 
                session_id=None, return_sequence=False, **kwargs):
        """
        Forward 函数
        :param session_id: 指定任务ID。如果为 None，则使用 set_active_head 设置的 ID。
        """
        # 1. 确定 Session ID
        if session_id is None:
            session_id = self.active_session_id
        
        if session_id is None:
            raise ValueError("TAM-CL requires session_id. Call set_active_head() or pass session_id to forward().")

        # 保存输入用于蒸馏 (如果处于训练模式)
        if self.training:
            self.last_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'image_tensor': image_tensor
            }

        # 2. 共享骨干输出序列特征 (B, L, H)
        # TAM-CL 需要序列特征来做 Attention
        seq = self.base_model(input_ids, attention_mask, token_type_ids, image_tensor,
                               return_sequence=True)
        
        B, L, H = seq.shape
        
        # 3. 获取任务 Token 并拼接 (B, 1, H)
        if session_id not in self.task_tokens:
             raise ValueError(f"Task token for {session_id} not found!")
        token = self.task_tokens[session_id].expand(B, -1, -1)
        seq_cat = torch.cat([token, seq], dim=1) # (B, L+1, H)
        
        # 4. 任务注意力块 (TAB)
        seq_tab = self.tab(seq_cat)  # (B, L+1, H)
        
        # 5. 取出任务 Token 的增强表示 (B, H)
        # 这是 TAB 融合了当前任务信息后的特征
        task_feat = seq_tab[:, 0, :]
        
        # 6. 分类头
        # [Fix 3] 使用 get_head 替代直接属性访问，防止 AttributeError
        head = self.head_manager.get_head(session_id)
        if head is None:
            raise ValueError(f"Head for session {session_id} not found in manager")
        
        # 获取任务类型
        # [Fix 3] 使用 get_task_name 替代直接属性访问
        task_name = self.head_manager.get_task_name(session_id)
        is_seq_task = task_name in ["mate", "mner", "mabsa"]
        
        if is_seq_task:
            # 序列任务：输入 TAB 输出的后续序列部分 (去掉 task token)
            logits = head(seq_tab[:, 1:, :]) 
        else:
            # 句级任务：输入增强后的 task token
            logits = head(task_feat)

        return logits, seq, seq_tab

    def compute_distillation(self, seq, session_id: str, T: float):
        """
        计算 KD Loss
        """
        if self.teacher is None:
            return torch.tensor(0.0).to(seq.device)            
        with torch.no_grad():
            # 教师模型跑一遍旧输入
            old_seq = self.teacher.base_model(
                self.last_inputs['input_ids'],
                self.last_inputs['attention_mask'],
                self.last_inputs.get('token_type_ids', None),
                self.last_inputs['image_tensor'],
                return_sequence=True
            )
            
        # 这里的 seq 是当前模型的 backbone 输出
        # 我们对 backbone 的输出做蒸馏，保证特征提取能力不退化
        B, L, H = seq.shape

        if old_seq.shape != seq.shape:
             return torch.tensor(0.0).to(seq.device)

        seq_flat = seq.reshape(-1, H)
        old_flat = old_seq.reshape(-1, H)
        
        log_q = F.log_softmax(seq_flat / T, dim=-1)
        p = F.softmax(old_flat / T, dim=-1)
        
        return F.kl_div(log_q, p, reduction='batchmean') * (T**2)

    def diversity_loss(self):
        """
        多样性损失：使不同任务的 Token 保持正交/远离
        """
        if len(self.task_tokens) <= 1:
            return torch.tensor(0.0).to(next(self.parameters()).device)
            
        tokens = torch.stack([t.squeeze(0) for t in self.task_tokens.values()])  # (n_sessions, H)
        
        # 计算余弦相似度矩阵
        sim = F.cosine_similarity(tokens.unsqueeze(1), tokens.unsqueeze(0), dim=-1)
        
        # 我们希望相似度尽可能小（接近0或负数）
        # 只取非对角线元素
        n = sim.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        
        # 最小化相似度 (即 maximize distance)
        # 这里的实现有多种变体，最简单的是惩罚相似度的平方或绝对值
        loss = sim[mask].pow(2).mean()
        
        return loss