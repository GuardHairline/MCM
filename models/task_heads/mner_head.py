# models/task_heads/mner_head.py
import torch
import torch.nn as nn
try:
    from torchcrf import CRF
    HAS_TORCHCRF = True
except ImportError:
    from models.task_heads.token_label_heads import SimpleCRF
    CRF = SimpleCRF
    HAS_TORCHCRF = False

class MNERHead(nn.Module):
    """
    [基础版] MNER任务头
    架构：
        Input (768) -> Dropout -> Linear (num_labels) -> CRF (Optional)
    """
    def __init__(self, input_dim, num_labels, dropout_prob=0.3, hidden_dim=None, use_crf=True):        
        super().__init__()
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.use_crf = use_crf
        
        self.dropout = nn.Dropout(dropout_prob)
        # 直接映射到 num_labels
        self.classifier = nn.Linear(input_dim, num_labels)
        
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)
        else:
            self.crf = None

    def forward(self, features, labels=None, mask=None):
        x = self.dropout(features)
        logits = self.classifier(x)
        
        if labels is not None:
            if self.use_crf:
                return self._compute_crf_loss(logits, labels, mask)
            else:
                return self._compute_ce_loss(logits, labels, mask)
        else:
            return logits

    def _compute_crf_loss(self, logits, labels, mask):
        if mask is None:
            mask = torch.ones(logits.size()[:2], dtype=torch.bool, device=logits.device)
        else:
            mask = mask.bool()
            
        crf_labels = labels.clone()
        crf_labels[crf_labels == -100] = 0
        
        # CRF Loss (Log Likelihood)
        # 注意：这里不再像 BiLSTM 版那样做复杂的 sequence 切割，
        # 因为 Transformer 输出通常已经对齐，且没有 LSTM 状态问题。
        # 但为了稳健，如果你想保持完全一致的 "Exclude CLS/SEP" 逻辑，也可以加上。
        # 这里使用标准 CRF 用法：
        log_likelihood = self.crf(logits, crf_labels, mask=mask, reduction='mean')
        return -log_likelihood, logits

    def _compute_ce_loss(self, logits, labels, mask):
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits
    def decode(self, features, mask=None):
            logits = self.forward(features, mask=mask, labels=None)
            if self.use_crf:
                if mask is None:
                    mask = torch.ones(logits.size()[:2], dtype=torch.bool, device=logits.device)
                else:
                    mask = mask.bool()
                preds_list = self.crf.decode(logits, mask=mask)
                # Pad to tensor
                bsz, seq_len, _ = logits.size()
                preds = torch.full((bsz, seq_len), -100, dtype=torch.long, device=logits.device)
                for i, p in enumerate(preds_list):
                    preds[i, :len(p)] = torch.tensor(p, device=logits.device)
                return preds
            else:
                return torch.argmax(logits, dim=-1)