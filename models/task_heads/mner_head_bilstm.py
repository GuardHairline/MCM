# models/task_heads/mner_head_bilstm.py
import torch
import torch.nn as nn

try:
    from torchcrf import CRF
    HAS_TORCHCRF = True
except ImportError:
    from models.task_heads.token_label_heads import SimpleCRF
    CRF = SimpleCRF
    HAS_TORCHCRF = False


class MNERHeadBiLSTM(nn.Module):
    """
    [增强版] MNER任务头 - 带 BiLSTM
    架构：
        Input -> Input Dropout -> BiLSTM -> Output Dropout -> Linear -> CRF
    
    参数：
        input_dim: 输入特征维度
        num_labels: 标签数量
        hidden_size: BiLSTM隐藏层大小
        num_lstm_layers: BiLSTM层数
        dropout: Dropout概率
        use_crf: 是否使用CRF层
    """
    
    def __init__(
        self, 
        input_dim: int = 768, 
        num_labels: int = 9,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        use_crf: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.use_crf = use_crf
        
        # 输入dropout
        self.input_dropout = nn.Dropout(dropout)
        
        # BiLSTM层
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        lstm_output_dim = hidden_size * 2

        # 输出dropout
        self.output_dropout = nn.Dropout(dropout)
        
        # 分类层
        self.classifier = nn.Linear(lstm_output_dim, num_labels)
        
        # CRF层（可选）
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)
        else:
            self.crf = None
    
    def forward(self, features, labels=None, mask=None):
        """
        前向传播
        
        参数:
            features: [batch_size, seq_len, input_dim] - 融合后的特征
            labels: [batch_size, seq_len] - 标签（训练时提供）
            mask: [batch_size, seq_len] - 注意力掩码
        
        返回:
            训练时：
                - 如果use_crf=True: (crf_loss, logits)
                - 如果use_crf=False: (ce_loss, logits)
            推理时：
                - logits: [batch_size, seq_len, num_labels]
        """
        # Input Dropout
        x = self.input_dropout(features)
        
        # BiLSTM
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            lengths = torch.clamp(lengths, min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.bilstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=features.size(1)
            )
        else:
            lstm_out, _ = self.bilstm(x)
            
        # Output Dropout
        x = self.output_dropout(lstm_out)
        
        # Linear
        logits = self.classifier(x)
        
        if labels is not None:
            if self.use_crf:
                return self._compute_crf_loss(logits, labels, mask)
            else:
                return self._compute_ce_loss(logits, labels, mask)
        return logits

    def _compute_crf_loss(self, logits, labels, mask):
        if mask is None:
            mask = torch.ones(logits.size()[:2], dtype=torch.bool, device=logits.device)
        else:
            mask = mask.bool()
        crf_labels = labels.clone()
        crf_labels[crf_labels == -100] = 0
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
            bsz, seq_len, _ = logits.size()
            preds = torch.full((bsz, seq_len), -100, dtype=torch.long, device=logits.device)
            for i, p in enumerate(preds_list):
                preds[i, :len(p)] = torch.tensor(p, device=logits.device)
            return preds
        else:
            return torch.argmax(logits, dim=-1)

