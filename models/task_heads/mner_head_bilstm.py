# models/task_heads/mner_head_bilstm.py
"""
改进的MNER Head - 添加BiLSTM层

基于早期MNER论文的成功经验，在DeBERTa输出和CRF之间添加BiLSTM层。
这样可以：
1. 提供双向序列建模能力
2. 避免DeBERTa的相对位置编码与CRF冲突
3. 增强序列标注性能

预期：F1从35%提升到55-60%
"""

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
    MNER任务头 - BiLSTM增强版
    
    架构：
        input_features (768) 
        → Dropout (0.3)
        → BiLSTM (256 hidden, 2 layers)
        → BiLSTM output (512)
        → Dropout (0.3)
        → Linear (num_labels)
        → CRF (可选)
    
    参数：
        input_dim: 输入特征维度（默认768，DeBERTa/ViT融合后的维度）
        num_labels: 标签数量（MNER通常为9：O + 4类实体的B/I标签）
        hidden_size: BiLSTM隐藏层大小（默认256）
        num_lstm_layers: BiLSTM层数（默认2）
        dropout: Dropout概率（默认0.3）
        use_crf: 是否使用CRF层（默认True）
    """
    
    def __init__(
        self, 
        input_dim: int = 768, 
        num_labels: int = 9,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        use_crf: bool = True,
        enable_bilstm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.use_crf = use_crf
        self.enable_bilstm = enable_bilstm
        
        # 输入dropout
        self.input_dropout = nn.Dropout(dropout)
        
        # BiLSTM层
        if enable_bilstm:
            self.bilstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_lstm_layers > 1 else 0.0
            )
            lstm_output_dim = hidden_size * 2
        else:
            self.bilstm = None
            lstm_output_dim = input_dim
            print("⚠️ MNERHeadBiLSTM: BiLSTM disabled, using direct projection.")
        
        # 输出dropout
        self.output_dropout = nn.Dropout(dropout)
        
        # 分类层
        self.classifier = nn.Linear(lstm_output_dim, num_labels)
        
        # CRF层（可选）
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)
            print(f"✓ MNERHeadBiLSTM initialized with CRF ({'torchcrf' if HAS_TORCHCRF else 'SimpleCRF'})")
        else:
            self.crf = None
            print("✓ MNERHeadBiLSTM initialized without CRF")
    
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
        batch_size, seq_len, _ = features.shape
        
        # 1. 输入dropout
        features = self.input_dropout(features)
        
        # 2. BiLSTM
        # 对于带padding的序列，使用pack_padded_sequence可以提高效率
        if self.enable_bilstm and mask is not None:
            lengths = mask.sum(dim=1).cpu()
            # 确保所有长度至少为1（避免pack_padded_sequence错误）
            lengths = torch.clamp(lengths, min=1)
            # Pack序列
            packed_features = nn.utils.rnn.pack_padded_sequence(
                features, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.bilstm(packed_features)
            # Unpack序列
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=seq_len
            )
            # 输出dropout
            lstm_output = self.output_dropout(lstm_output)
        elif self.enable_bilstm:
            lstm_output, _ = self.bilstm(features)
            # 输出dropout
            lstm_output = self.output_dropout(lstm_output)
        else:
            lstm_output = features
        
        
        # 4. 分类
        logits = self.classifier(lstm_output)  # [batch_size, seq_len, num_labels]
        
        # 5. 训练或推理
        if labels is not None:
            # 训练模式
            if self.use_crf:
                # 使用CRF loss
                return self._compute_crf_loss(logits, labels, mask)
            else:
                # 使用交叉熵loss
                return self._compute_ce_loss(logits, labels, mask)
        else:
            # 推理模式：返回logits
            return logits
    
    def _compute_crf_loss(self, logits, labels, mask):
        """计算CRF loss"""
        # 准备mask和labels
        if mask is None:
            mask = torch.ones(logits.size()[:2], dtype=torch.bool, device=logits.device)
        else:
            mask = mask.bool()
        
        # 处理labels中的-100（padding标记）
        crf_labels = labels.clone()
        crf_labels[crf_labels == -100] = 0  # CRF不支持-100，临时替换为0
        
        # 提取有效token范围（排除[CLS]和[SEP]）
        # 假设：[CLS] token在位置0，[SEP] token在最后
        valid_logits_list = []
        valid_labels_list = []
        valid_mask_list = []
        
        batch_size = logits.size(0)
        for i in range(batch_size):
            # 找到有效token的范围（mask为True的部分）
            valid_indices = mask[i].nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                # 排除第一个token ([CLS]) 和最后一个token ([SEP])
                if len(valid_indices) > 2:
                    start_idx = valid_indices[0] + 1  # 跳过[CLS]
                    end_idx = valid_indices[-1]       # 不包括[SEP]
                    
                    valid_logits_list.append(logits[i, start_idx:end_idx])
                    valid_labels_list.append(crf_labels[i, start_idx:end_idx])
                    valid_len = end_idx - start_idx
                    valid_mask_list.append(torch.ones(valid_len, dtype=torch.bool, device=logits.device))
                else:
                    # 序列太短，使用全部token
                    valid_logits_list.append(logits[i, :len(valid_indices)])
                    valid_labels_list.append(crf_labels[i, :len(valid_indices)])
                    valid_mask_list.append(torch.ones(len(valid_indices), dtype=torch.bool, device=logits.device))
            else:
                # 空序列，添加dummy
                valid_logits_list.append(logits[i, :1])
                valid_labels_list.append(crf_labels[i, :1])
                valid_mask_list.append(torch.ones(1, dtype=torch.bool, device=logits.device))
        
        # Pad到相同长度
        max_len = max(v.size(0) for v in valid_logits_list)
        padded_logits = torch.zeros(batch_size, max_len, self.num_labels, device=logits.device)
        padded_labels = torch.zeros(batch_size, max_len, dtype=torch.long, device=logits.device)
        padded_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=logits.device)
        
        for i in range(batch_size):
            valid_len = valid_logits_list[i].size(0)
            padded_logits[i, :valid_len] = valid_logits_list[i]
            padded_labels[i, :valid_len] = valid_labels_list[i]
            padded_mask[i, :valid_len] = valid_mask_list[i]
        
        # 计算CRF loss
        # torchcrf返回的是log likelihood，需要取负数作为loss
        log_likelihood = self.crf(padded_logits, padded_labels, mask=padded_mask, reduction='mean')
        loss = -log_likelihood
        
        return loss, logits
    
    def _compute_ce_loss(self, logits, labels, mask):
        """计算交叉熵loss"""
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 将logits和labels展平
        active_loss = mask.view(-1) == 1 if mask is not None else None
        
        if active_loss is not None:
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return loss, logits
    
    def decode(self, features, mask=None):
        """
        解码：使用Viterbi算法（如果有CRF）或argmax
        
        参数:
            features: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len]
        
        返回:
            predictions: [batch_size, seq_len] - 预测的标签ID
        """
        # 获取logits
        logits = self.forward(features, mask=mask, labels=None)
        
        if self.use_crf:
            # 使用CRF的Viterbi解码
            if mask is None:
                crf_mask = torch.ones(logits.size()[:2], dtype=torch.bool, device=logits.device)
            else:
                crf_mask = mask.bool()
            
            # 提取有效token范围
            batch_size = logits.size(0)
            predictions = torch.zeros(logits.size()[:2], dtype=torch.long, device=logits.device)
            
            for i in range(batch_size):
                valid_indices = crf_mask[i].nonzero(as_tuple=True)[0]
                if len(valid_indices) > 2:
                    start_idx = valid_indices[0] + 1
                    end_idx = valid_indices[-1]
                    
                    valid_logits = logits[i, start_idx:end_idx].unsqueeze(0)
                    valid_mask = torch.ones(1, end_idx - start_idx, dtype=torch.bool, device=logits.device)
                    
                    valid_preds = self.crf.decode(valid_logits, mask=valid_mask)[0]
                    predictions[i, start_idx:end_idx] = torch.tensor(valid_preds, device=logits.device)
                else:
                    # 使用argmax
                    predictions[i] = torch.argmax(logits[i], dim=-1)
        else:
            # 使用argmax解码
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions


def test_mner_head_bilstm():
    """测试MNERHeadBiLSTM"""
    print("="*80)
    print("测试 MNERHeadBiLSTM")
    print("="*80)
    
    # 参数
    batch_size = 4
    seq_len = 32
    input_dim = 768
    num_labels = 9
    
    # 创建模型
    print("\n1. 创建模型（使用CRF）...")
    head_with_crf = MNERHeadBiLSTM(
        input_dim=input_dim,
        num_labels=num_labels,
        hidden_size=256,
        num_lstm_layers=2,
        dropout=0.3,
        use_crf=True
    )
    print(f"   参数量: {sum(p.numel() for p in head_with_crf.parameters()):,}")
    
    # 模拟输入
    print("\n2. 模拟输入...")
    features = torch.randn(batch_size, seq_len, input_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    # 最后几个位置是padding
    attention_mask[:, -5:] = 0
    
    labels = torch.randint(0, num_labels, (batch_size, seq_len))
    labels[attention_mask == 0] = -100
    
    print(f"   features: {features.shape}")
    print(f"   attention_mask: {attention_mask.shape}")
    print(f"   labels: {labels.shape}")
    
    # 测试训练
    print("\n3. 测试训练（forward with labels）...")
    loss, logits = head_with_crf(features, attention_mask, labels)
    print(f"   ✓ Loss: {loss.item():.4f}")
    print(f"   ✓ Logits shape: {logits.shape}")
    
    # 测试推理
    print("\n4. 测试推理（forward without labels）...")
    with torch.no_grad():
        logits = head_with_crf(features, attention_mask)
        print(f"   ✓ Logits shape: {logits.shape}")
    
    # 测试解码
    print("\n5. 测试解码（Viterbi）...")
    with torch.no_grad():
        predictions = head_with_crf.decode(features, attention_mask)
        print(f"   ✓ Predictions shape: {predictions.shape}")
        print(f"   ✓ Predictions sample: {predictions[0, :10].tolist()}")
    
    # 测试不使用CRF
    print("\n6. 创建模型（不使用CRF）...")
    head_no_crf = MNERHeadBiLSTM(
        input_dim=input_dim,
        num_labels=num_labels,
        use_crf=False
    )
    loss, logits = head_no_crf(features, attention_mask, labels)
    print(f"   ✓ Loss: {loss.item():.4f}")
    
    print("\n" + "="*80)
    print("✅ 所有测试通过！")
    print("="*80)


if __name__ == "__main__":
    test_mner_head_bilstm()

