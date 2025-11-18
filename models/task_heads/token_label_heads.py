import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入CRF，如果没有则使用简化版本
try:
    from torchcrf import CRF
    HAS_TORCHCRF = True
except ImportError:
    HAS_TORCHCRF = False
    print("Warning: torchcrf not found, using simplified CRF implementation")


class SimpleCRF(nn.Module):
    """简化的CRF实现（当torchcrf不可用时）"""
    def __init__(self, num_labels, batch_first=True):
        super().__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        # 转移矩阵: transitions[i,j] = 从标签i转移到标签j的分数
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        # 起始和结束转移
        self.start_transitions = nn.Parameter(torch.randn(num_labels))
        self.end_transitions = nn.Parameter(torch.randn(num_labels))
    
    def forward(self, emissions, tags, mask=None, reduction='mean'):
        """
        计算log likelihood（与torchcrf.CRF一致）
        
        注意：返回的是log likelihood（不是NLL），越大越好
        如果要作为loss使用，需要取负值
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        batch_size, seq_len, num_labels = emissions.size()
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=emissions.device)
        
        # 计算分数
        score = self._compute_score(emissions, tags, mask)
        # 计算配分函数
        partition = self._compute_normalizer(emissions, mask)
        
        # log likelihood = score - partition
        # (与torchcrf一致，返回log likelihood而不是NLL)
        log_likelihood = score - partition
        
        if reduction == 'mean':
            return log_likelihood.mean()
        elif reduction == 'sum':
            return log_likelihood.sum()
        else:
            return log_likelihood
    
    def decode(self, emissions, mask=None):
        """Viterbi解码"""
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        batch_size, seq_len, num_labels = emissions.size()
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=emissions.device)
        
        # Viterbi算法
        best_paths = []
        for i in range(batch_size):
            seq_emissions = emissions[i]
            seq_mask = mask[i]
            seq_len_i = seq_mask.sum().item()
            
            # 初始化
            viterbi = seq_emissions[0] + self.start_transitions
            backpointers = []
            
            # 前向传播
            for t in range(1, seq_len_i):
                next_viterbi = viterbi.unsqueeze(1) + self.transitions
                next_viterbi = next_viterbi + seq_emissions[t]
                viterbi, best_tags = next_viterbi.max(dim=0)
                backpointers.append(best_tags)
            
            # 添加结束转移
            viterbi = viterbi + self.end_transitions
            best_last_tag = viterbi.argmax()
            
            # 回溯
            best_path = [best_last_tag.item()]
            for bp in reversed(backpointers):
                best_path.append(bp[best_path[-1]].item())
            best_path.reverse()
            
            # 填充到完整长度
            best_path = best_path + [0] * (seq_len - seq_len_i)
            best_paths.append(best_path)
        
        return best_paths
    
    def _compute_score(self, emissions, tags, mask):
        batch_size, seq_len = tags.size()
        scores = torch.zeros(batch_size, device=emissions.device)
        
        for i in range(batch_size):
            seq_len_i = mask[i].sum().item()
            # 起始转移
            scores[i] += self.start_transitions[tags[i, 0]]
            # 发射分数
            scores[i] += emissions[i, 0, tags[i, 0]]
            
            # 转移分数
            for t in range(1, seq_len_i):
                scores[i] += self.transitions[tags[i, t-1], tags[i, t]]
                scores[i] += emissions[i, t, tags[i, t]]
            
            # 结束转移
            scores[i] += self.end_transitions[tags[i, seq_len_i-1]]
        
        return scores
    
    def _compute_normalizer(self, emissions, mask):
        batch_size, seq_len, num_labels = emissions.size()
        
        # 初始化alpha
        alpha = emissions[:, 0] + self.start_transitions.unsqueeze(0)
        
        for t in range(1, seq_len):
            # alpha: (batch, num_labels)
            # transitions: (num_labels, num_labels)
            # emissions: (batch, seq_len, num_labels)
            
            # broadcast: (batch, num_labels, 1) + (num_labels, num_labels)
            next_alpha = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            next_alpha = next_alpha + emissions[:, t].unsqueeze(1)
            
            # log-sum-exp
            next_alpha = torch.logsumexp(next_alpha, dim=1)
            
            # 应用mask
            alpha = torch.where(mask[:, t].unsqueeze(1), next_alpha, alpha)
        
        # 添加结束转移
        alpha = alpha + self.end_transitions.unsqueeze(0)
        
        # 最终的配分函数
        return torch.logsumexp(alpha, dim=1)


class TokenLabelHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, label_emb, task_name, use_crf=True):
        super().__init__()
        self.num_labels = num_labels
        self.label_emb = label_emb  # GlobalLabelEmbedding 实例
        self.task_name = task_name
        self.use_crf = use_crf and num_labels > 2  # 只对多类别任务使用CRF

        # 将 token 向量投影到与 label 相同维度
        self.token_proj = nn.Linear(input_dim, hidden_dim)
        self.label_proj = nn.Linear(label_emb.emb_dim, hidden_dim)

        # 可选：归一化分母
        self.scale = hidden_dim ** 0.5
        
        # 添加CRF层
        if self.use_crf:
            if HAS_TORCHCRF:
                self.crf = CRF(num_labels, batch_first=True)
            else:
                self.crf = SimpleCRF(num_labels, batch_first=True)
            print(f"[{task_name}] TokenLabelHead initialized with CRF (num_labels={num_labels})")
        else:
            self.crf = None
            print(f"[{task_name}] TokenLabelHead initialized without CRF (num_labels={num_labels})")

    def forward(self, seq_feats, labels=None, mask=None):
        """
        Args:
            seq_feats: (batch, seq_len, input_dim)
            labels: (batch, seq_len) - 仅在训练时提供
            mask: (batch, seq_len) - 有效token的mask
        
        Returns:
            如果training且use_crf: (loss, logits)
            否则: logits or predictions
        """
        # seq_feats: (batch, seq_len, input_dim)
        token_h = self.token_proj(seq_feats)  # (B, L, H)

        # 获取该任务所有 label 的 embedding 并做投影
        label_ids = torch.arange(self.num_labels, device=seq_feats.device)
        label_embs = self.label_emb(self.task_name, label_ids)   # (num_labels, emb_dim)
        label_h = self.label_proj(label_embs)                    # (num_labels, H)

        # 使用 dot product 得到 logits
        logits = torch.matmul(token_h, label_h.T) / self.scale   # (B, L, num_labels)

        # 如果使用CRF
        if self.use_crf and self.crf is not None:
            if labels is not None:
                # 训练模式：计算CRF loss
                # 创建mask（如果没有提供）
                if mask is None:
                    mask = (labels != -100)
                
                # 将-100替换为0（CRF不支持-100）
                crf_labels = labels.clone()
                crf_labels[~mask] = 0
                
                # 计算negative log likelihood
                nll = self.crf(logits, crf_labels, mask=mask, reduction='mean')
                return nll, logits
            else:
                # 评估模式：返回logits
                # 注意：虽然CRF的decode能得到全局最优解，但为了与现有评估流程兼容，
                # 这里返回logits，让evaluate.py统一处理
                return logits
        
        # 不使用CRF：直接返回logits
        return logits
