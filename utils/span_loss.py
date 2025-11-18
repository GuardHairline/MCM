"""
Span-level loss functions for sequence labeling tasks

这些loss函数关注整个实体span的正确性，而不仅仅是token级别的准确性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Set, Tuple
from utils.decode import decode_mate, decode_mner, decode_mabsa


def compute_span_f1_loss(logits: torch.Tensor, 
                         labels: torch.Tensor,
                         task_name: str,
                         reduction: str = 'mean') -> torch.Tensor:
    """
    计算基于span F1的loss
    
    Args:
        logits: (batch_size, seq_len, num_labels) - 模型输出的logits
        labels: (batch_size, seq_len) - 真实标签
        task_name: 任务名称 (mate, mner, mabsa)
        reduction: 'mean' or 'sum'
    
    Returns:
        span_loss: 标量loss，值越小越好
    
    原理：
        - 将logits转换为预测标签
        - 解码成chunks/spans
        - 计算F1分数
        - 返回 (1 - F1) 作为loss
    """
    # 选择解码函数
    decode_fn = {
        'mate': decode_mate,
        'mner': decode_mner,
        'mabsa': decode_mabsa
    }.get(task_name, decode_mate)
    
    batch_size = logits.size(0)
    device = logits.device
    
    # 获取预测标签
    preds = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
    
    # 对每个样本计算span F1
    f1_scores = []
    
    for i in range(batch_size):
        # 获取有效长度
        valid_mask = (labels[i] != -100)
        valid_len = valid_mask.sum().item()
        
        if valid_len == 0:
            # 跳过完全padding的样本
            continue
        
        # 提取有效部分
        pred_seq = preds[i, :valid_len].cpu().tolist()
        gold_seq = labels[i, :valid_len].cpu().tolist()
        
        # 解码成chunks
        try:
            pred_chunks = decode_fn(pred_seq)
            gold_chunks = decode_fn(gold_seq)
        except Exception as e:
            # 解码失败时，使用0分数
            f1_scores.append(0.0)
            continue
        
        # 计算F1
        if len(gold_chunks) == 0 and len(pred_chunks) == 0:
            # 都为空：完美匹配
            f1 = 1.0
        elif len(gold_chunks) == 0 or len(pred_chunks) == 0:
            # 一个为空一个不为空：完全错误
            f1 = 0.0
        else:
            # 计算TP, FP, FN
            tp = len(pred_chunks & gold_chunks)
            fp = len(pred_chunks - gold_chunks)
            fn = len(gold_chunks - pred_chunks)
            
            # F1 = 2*TP / (2*TP + FP + FN)
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        f1_scores.append(f1)
    
    # 转换为tensor
    if len(f1_scores) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    f1_tensor = torch.tensor(f1_scores, device=device)
    
    # Loss = 1 - F1
    loss = 1.0 - f1_tensor
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def compute_boundary_loss(logits: torch.Tensor,
                         labels: torch.Tensor,
                         task_name: str,
                         reduction: str = 'mean') -> torch.Tensor:
    """
    计算边界检测loss
    
    专注于实体的开始位置(B)和结束位置的检测
    
    Args:
        logits: (batch_size, seq_len, num_labels)
        labels: (batch_size, seq_len)
        task_name: 任务名称
        reduction: 'mean' or 'sum'
    
    Returns:
        boundary_loss: 边界检测loss
    
    原理：
        - 对B标签给予更高的权重
        - 惩罚B→I的错误转移
        - 惩罚O→I的非法转移
    """
    batch_size, seq_len, num_labels = logits.shape
    device = logits.device
    
    # 定义标签权重（根据任务调整）
    if task_name == 'mate':
        # O=0, B=1, I=2
        # 给B标签更高权重
        label_weights = torch.tensor([1.0, 3.0, 1.5], device=device)
    elif task_name == 'mner':
        # O=0, B-type=奇数, I-type=偶数
        # 给所有B标签更高权重
        label_weights = torch.ones(num_labels, device=device)
        for i in range(1, num_labels, 2):  # B标签是奇数
            label_weights[i] = 3.0
        for i in range(2, num_labels, 2):  # I标签是偶数
            label_weights[i] = 1.5
    elif task_name == 'mabsa':
        # O=0, B-*=1,3,5, I-*=2,4,6
        label_weights = torch.ones(num_labels, device=device)
        for i in [1, 3, 5]:  # B标签
            label_weights[i] = 3.0
        for i in [2, 4, 6]:  # I标签
            label_weights[i] = 1.5
    else:
        label_weights = torch.ones(num_labels, device=device)
    
    # 计算加权的CrossEntropy loss
    loss = F.cross_entropy(
        logits.reshape(-1, num_labels),
        labels.reshape(-1),
        weight=label_weights,
        ignore_index=-100,
        reduction=reduction
    )
    
    return loss


def compute_transition_penalty(logits: torch.Tensor,
                               labels: torch.Tensor,
                               task_name: str,
                               reduction: str = 'mean') -> torch.Tensor:
    """
    计算非法转移的惩罚
    
    惩罚违反BIO约束的转移，例如：
    - O → I (I必须跟在B后面)
    - B-type1 → I-type2 (实体类型不一致)
    
    Args:
        logits: (batch_size, seq_len, num_labels)
        labels: (batch_size, seq_len)
        task_name: 任务名称
        reduction: 'mean' or 'sum'
    
    Returns:
        transition_penalty: 转移惩罚loss
    """
    batch_size, seq_len, num_labels = logits.shape
    device = logits.device
    
    # 获取预测标签
    preds = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
    
    # 计算非法转移的数量
    penalties = []
    
    for i in range(batch_size):
        valid_mask = (labels[i] != -100)
        valid_len = valid_mask.sum().item()
        
        if valid_len <= 1:
            continue
        
        pred_seq = preds[i, :valid_len]
        
        penalty = 0.0
        
        # 检查每个转移
        for t in range(valid_len - 1):
            prev_label = pred_seq[t].item()
            curr_label = pred_seq[t+1].item()
            
            # 检查非法转移（根据任务）
            if task_name == 'mate':
                # O=0, B=1, I=2
                # 非法：O→I
                if prev_label == 0 and curr_label == 2:
                    penalty += 1.0
            
            elif task_name == 'mner':
                # 非法：O→I-type, B-type1→I-type2
                if prev_label == 0 and curr_label % 2 == 0:  # O→I
                    penalty += 1.0
                elif prev_label % 2 == 1 and curr_label % 2 == 0:  # B→I
                    # 检查类型是否匹配
                    prev_type = (prev_label - 1) // 2
                    curr_type = (curr_label - 2) // 2
                    if prev_type != curr_type:
                        penalty += 1.0
            
            elif task_name == 'mabsa':
                # 非法：O→I-*, B-sent1→I-sent2
                if prev_label == 0 and curr_label % 2 == 0:  # O→I
                    penalty += 1.0
                elif prev_label % 2 == 1 and curr_label % 2 == 0:  # B→I
                    # 检查情感类型是否匹配
                    if prev_label in [1, 2] and curr_label not in [1, 2]:
                        penalty += 1.0
                    elif prev_label in [3, 4] and curr_label not in [3, 4]:
                        penalty += 1.0
                    elif prev_label in [5, 6] and curr_label not in [5, 6]:
                        penalty += 1.0
        
        # 归一化
        if valid_len > 1:
            penalty = penalty / (valid_len - 1)
        
        penalties.append(penalty)
    
    if len(penalties) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    penalty_tensor = torch.tensor(penalties, device=device)
    
    if reduction == 'mean':
        return penalty_tensor.mean()
    elif reduction == 'sum':
        return penalty_tensor.sum()
    else:
        return penalty_tensor


class SpanLoss(nn.Module):
    """
    综合的Span Loss模块
    
    组合多个span-level的loss：
    1. Span F1 Loss: 直接优化span级别的F1
    2. Boundary Loss: 加强边界检测
    3. Transition Penalty: 惩罚非法转移
    """
    def __init__(self, 
                 task_name: str,
                 span_f1_weight: float = 0.3,
                 boundary_weight: float = 0.1,
                 transition_weight: float = 0.1):
        super().__init__()
        self.task_name = task_name
        self.span_f1_weight = span_f1_weight
        self.boundary_weight = boundary_weight
        self.transition_weight = transition_weight
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算综合span loss
        
        Args:
            logits: (batch_size, seq_len, num_labels)
            labels: (batch_size, seq_len)
        
        Returns:
            total_loss: 综合loss
        """
        loss = 0.0
        
        # 1. Span F1 Loss (可微分的近似)
        if self.span_f1_weight > 0:
            try:
                span_f1_loss = compute_span_f1_loss(
                    logits, labels, self.task_name
                )
                # 注意：span_f1_loss基于argmax，不可微
                # 这里仅用于监控，不参与梯度计算
                # loss += self.span_f1_weight * span_f1_loss.detach()
            except Exception:
                span_f1_loss = torch.tensor(0.0)
        
        # 2. Boundary Loss (可微分)
        if self.boundary_weight > 0:
            boundary_loss = compute_boundary_loss(
                logits, labels, self.task_name
            )
            loss += self.boundary_weight * boundary_loss
        
        # 3. Transition Penalty (基于argmax，不可微)
        # if self.transition_weight > 0:
        #     transition_loss = compute_transition_penalty(
        #         logits, labels, self.task_name
        #     )
        #     loss += self.transition_weight * transition_loss.detach()
        
        return loss if isinstance(loss, torch.Tensor) else torch.tensor(0.0, device=logits.device)

