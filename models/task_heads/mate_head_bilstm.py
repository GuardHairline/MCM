# models/task_heads/mate_head_bilstm.py
"""
MATE任务头 - BiLSTM增强版

基于 simple_ner_training.py 的成功经验（~80% F1），添加BiLSTM层。

架构：
    input_features (768)
    → Dropout (0.3)
    → BiLSTM (256 hidden, 2 layers)
    → BiLSTM output (512)
    → Dropout (0.3)
    → Linear (num_labels=3)
    → CRF (可选)

预期：性能提升 10-15%
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


class MATEHeadBiLSTM(nn.Module):
    """
    MATE (Multimodal Aspect Term Extraction) 任务头 - BiLSTM增强版
    
    标签: O=0, B=1, I=2 (3个标签)
    
    参数:
        input_dim: 输入特征维度（默认768）
        num_labels: 标签数量（MATE固定为3）
        hidden_size: BiLSTM隐藏层大小（默认256）
        num_lstm_layers: BiLSTM层数（默认2）
        dropout: Dropout概率（默认0.3）
        use_crf: 是否使用CRF层（默认True）
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_labels: int = 3,  # O, B, I
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
        self.use_crf = use_crf and num_labels > 2
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
            print("⚠️ MATEHeadBiLSTM: BiLSTM disabled, using direct projection.")
        
        # 输出dropout
        self.output_dropout = nn.Dropout(dropout)
        
        # 分类层
        self.classifier = nn.Linear(lstm_output_dim, num_labels)
        
        # CRF层（可选）
        if self.use_crf:
            self.crf = CRF(num_labels, batch_first=True)
            print(f"✓ MATEHeadBiLSTM initialized with CRF ({'torchcrf' if HAS_TORCHCRF else 'SimpleCRF'})")
        else:
            self.crf = None
            print("✓ MATEHeadBiLSTM initialized without CRF")
    
    def forward(self, seq_features, labels=None, mask=None):
        """
        前向传播
        
        参数:
            seq_features: [batch_size, seq_len, input_dim] - 融合后的特征
            labels: [batch_size, seq_len] - 标签（训练时提供）
            mask: [batch_size, seq_len] - 注意力掩码
        
        返回:
            训练时：(loss, logits)
            推理时：logits
        """
        batch_size, seq_len, _ = seq_features.shape
        
        # 1. 输入dropout
        features = self.input_dropout(seq_features)
        
        # 2. BiLSTM
        if self.enable_bilstm and mask is not None:
            lengths = mask.sum(dim=1).cpu()
            # 确保所有长度至少为1（避免pack_padded_sequence错误）
            lengths = torch.clamp(lengths, min=1)
            packed_features = nn.utils.rnn.pack_padded_sequence(
                features, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.bilstm(packed_features)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=seq_len
            )
        elif self.enable_bilstm:
            lstm_output, _ = self.bilstm(features)
        else:
            lstm_output = features
        
        # 3. 输出dropout
        lstm_output = self.output_dropout(lstm_output)
        
        # 4. 分类
        logits = self.classifier(lstm_output)  # [batch_size, seq_len, num_labels]
        
        # 5. 训练或推理
        if labels is not None:
            if self.use_crf:
                return self._compute_crf_loss(logits, labels, mask)
            else:
                return self._compute_ce_loss(logits, labels, mask)
        else:
            return logits
    
    def _compute_crf_loss(self, logits, labels, mask):
        """计算CRF loss"""
        batch_size = logits.size(0)
        crf_loss_total = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            # 找到有效token的范围（label != -100）
            valid_mask = (labels[i] != -100)
            if valid_mask.any():
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                start_idx = valid_indices[0].item()
                end_idx = valid_indices[-1].item() + 1
                
                # 提取有效范围
                sample_logits = logits[i:i+1, start_idx:end_idx, :]
                sample_labels = labels[i:i+1, start_idx:end_idx]
                sample_mask = torch.ones(
                    1, end_idx - start_idx,
                    dtype=torch.bool,
                    device=logits.device
                )
                
                # CRF forward：返回log likelihood
                log_likelihood = self.crf(
                    sample_logits, sample_labels,
                    mask=sample_mask, reduction='sum'
                )
                crf_loss_total += -log_likelihood  # 转换为NLL
                valid_samples += 1
        
        # 平均loss
        if valid_samples > 0:
            nll = crf_loss_total / valid_samples
        else:
            nll = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return nll, logits
    
    def _compute_ce_loss(self, logits, labels, mask):
        """计算交叉熵loss"""
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        if mask is not None:
            active_loss = mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return loss, logits
    
    def decode(self, seq_features, mask=None):
        """
        解码：使用Viterbi算法（如果有CRF）或argmax
        
        参数:
            seq_features: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len]
        
        返回:
            predictions: [batch_size, seq_len] - 预测的标签ID
        """
        logits = self.forward(seq_features, labels=None, mask=mask)
        
        if self.use_crf:
            # CRF Viterbi解码
            if mask is None:
                mask = torch.ones(logits.size()[:2], dtype=torch.bool, device=logits.device)
            else:
                mask = mask.bool()
            
            batch_size = logits.size(0)
            predictions = torch.zeros(logits.size()[:2], dtype=torch.long, device=logits.device)
            
            for i in range(batch_size):
                valid_indices = mask[i].nonzero(as_tuple=True)[0]
                if len(valid_indices) > 2:
                    start_idx = valid_indices[0] + 1  # 跳过[CLS]
                    end_idx = valid_indices[-1]  # 不包括[SEP]
                    
                    valid_logits = logits[i, start_idx:end_idx].unsqueeze(0)
                    valid_mask = torch.ones(
                        1, end_idx - start_idx,
                        dtype=torch.bool,
                        device=logits.device
                    )
                    
                    valid_preds = self.crf.decode(valid_logits, mask=valid_mask)[0]
                    predictions[i, start_idx:end_idx] = torch.tensor(
                        valid_preds, device=logits.device
                    )
                else:
                    predictions[i] = torch.argmax(logits[i], dim=-1)
        else:
            # Argmax解码
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions


def test_mate_head_bilstm():
    """测试MATEHeadBiLSTM"""
    print("="*80)
    print("测试 MATEHeadBiLSTM")
    print("="*80)
    
    # 参数
    batch_size = 4
    seq_len = 32
    input_dim = 768
    num_labels = 3  # O, B, I
    
    # 创建模型
    print("\n1. 创建模型（使用CRF）...")
    head_with_crf = MATEHeadBiLSTM(
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
    mask = torch.ones(batch_size, seq_len)
    mask[:, -5:] = 0  # 最后几个位置是padding
    
    labels = torch.randint(0, num_labels, (batch_size, seq_len))
    labels[mask == 0] = -100
    
    print(f"   features: {features.shape}")
    print(f"   mask: {mask.shape}")
    print(f"   labels: {labels.shape}")
    
    # 测试训练
    print("\n3. 测试训练（forward with labels）...")
    loss, logits = head_with_crf(features, labels, mask)
    print(f"   ✓ Loss: {loss.item():.4f}")
    print(f"   ✓ Logits shape: {logits.shape}")
    
    # 测试推理
    print("\n4. 测试推理（forward without labels）...")
    with torch.no_grad():
        logits = head_with_crf(features, mask=mask)
        print(f"   ✓ Logits shape: {logits.shape}")
    
    # 测试解码
    print("\n5. 测试解码（Viterbi）...")
    with torch.no_grad():
        predictions = head_with_crf.decode(features, mask)
        print(f"   ✓ Predictions shape: {predictions.shape}")
        print(f"   ✓ Predictions sample: {predictions[0, :10].tolist()}")
    
    # 测试不使用CRF
    print("\n6. 创建模型（不使用CRF）...")
    head_no_crf = MATEHeadBiLSTM(
        input_dim=input_dim,
        num_labels=num_labels,
        use_crf=False
    )
    loss, logits = head_no_crf(features, labels, mask)
    print(f"   ✓ Loss: {loss.item():.4f}")
    
    # 反向传播测试
    print("\n7. 测试反向传播...")
    optimizer = torch.optim.Adam(head_with_crf.parameters(), lr=1e-4)
    loss, _ = head_with_crf(features, labels, mask)
    optimizer.zero_grad()
    loss.backward()
    
    has_grad = False
    for name, param in head_with_crf.named_parameters():
        if param.grad is not None:
            has_grad = True
            assert not torch.isnan(param.grad).any(), f"❌ {name} has NaN gradients"
            assert not torch.isinf(param.grad).any(), f"❌ {name} has Inf gradients"
    
    assert has_grad, "❌ No parameters have gradients"
    print(f"   ✓ 反向传播成功，梯度正常")
    
    optimizer.step()
    print(f"   ✓ 参数更新成功")
    
    print("\n" + "="*80)
    print("✅ 所有测试通过！")
    print("="*80)


if __name__ == "__main__":
    test_mate_head_bilstm()

