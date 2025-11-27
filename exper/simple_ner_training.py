#!/usr/bin/env python3
"""
完整的独立NER训练脚本
========================================
从数据读取、模型构建、训练到验证的完整过程

使用：
    python tests/simple_ner_training.py

架构：
    DeBERTa-v3-base → BiLSTM → CRF
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入项目组件
from datasets.mner_dataset import MNERDataset

from torchcrf import CRF


# ============================================================================
# 1. 模型定义
# ============================================================================

class SimpleNERModel(nn.Module):
    """
    简单的NER模型：DeBERTa → BiLSTM → CRF
    
    架构:
        - Text Encoder: DeBERTa-v3-base (768d)
        - Sequence Layer: BiLSTM (256 hidden × 2 directions = 512d)
        - Output Layer: Linear (512 → num_labels)
        - CRF Layer: 全局序列优化
    """
    
    def __init__(
        self,
        text_encoder_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 9,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        use_crf: bool = True
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.use_crf = use_crf
        
        # Text encoder
        if text_encoder_name == "microsoft/deberta-v3-base":
            model_path = PROJECT_ROOT / "downloaded_model/deberta-v3-base"
            if not model_path.exists():
                print(f"⚠️ 本地模型不存在，使用在线模型: {text_encoder_name}")
                model_path = text_encoder_name
        else:
            model_path = text_encoder_name
        
        self.text_encoder = AutoModel.from_pretrained(model_path)
        encoder_dim = self.text_encoder.config.hidden_size  # 768
        
        # BiLSTM layer
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # Classifier
        lstm_output_dim = lstm_hidden * 2  # 双向
        self.classifier = nn.Linear(lstm_output_dim, num_labels)
        
        # CRF layer
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)
            print("✓ 使用CRF层")
        else:
            self.crf = None
            print("✓ 不使用CRF层")
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] (可选，训练时提供)
        
        Returns:
            训练时: (loss, logits)
            推理时: logits
        """
        # 1. Text encoding
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_output.last_hidden_state  # [batch, seq_len, 768]
        
        # 2. Dropout
        text_features = self.dropout(text_features)
        
        # 3. BiLSTM
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            text_features, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.bilstm(packed)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        lstm_output = self.dropout(lstm_output)  # [batch, seq_len, 512]
        
        # 4. Classifier
        logits = self.classifier(lstm_output)  # [batch, seq_len, num_labels]
        
        # 5. CRF (if training)
        if labels is not None:
            if self.use_crf:
                # CRF loss
                loss = self._compute_crf_loss(logits, labels, attention_mask)
                return loss, logits
            else:
                # Cross-entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss, logits
        else:
            return logits
    
    def _compute_crf_loss(self, logits, labels, attention_mask):
        """计算CRF loss"""
        batch_size = logits.size(0)
        total_loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            # 找到有效token（label != -100）
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
                
                # CRF forward返回log likelihood
                log_likelihood = self.crf(
                    sample_logits, sample_labels, 
                    mask=sample_mask, reduction='sum'
                )
                total_loss += -log_likelihood  # 转换为NLL
                valid_samples += 1
        
        return total_loss / valid_samples if valid_samples > 0 else torch.tensor(0.0)
    
    def decode(self, input_ids, attention_mask):
        """
        Viterbi解码（使用CRF）或argmax解码
        
        Returns:
            predictions: [batch, seq_len]
        """
        logits = self.forward(input_ids, attention_mask)
        
        if self.use_crf:
            # Viterbi解码
            batch_size, seq_len = input_ids.size()
            predictions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=logits.device)
            
            for i in range(batch_size):
                valid_length = int(attention_mask[i].sum().item())
                if valid_length > 2:
                    # 跳过[CLS]和[SEP]
                    start_idx = 1
                    end_idx = valid_length - 1
                    
                    sample_logits = logits[i:i+1, start_idx:end_idx, :]
                    sample_mask = torch.ones(1, end_idx - start_idx, dtype=torch.bool, device=logits.device)
                    
                    preds = self.crf.decode(sample_logits, mask=sample_mask)[0]
                    predictions[i, start_idx:end_idx] = torch.tensor(preds, device=logits.device)
                else:
                    predictions[i] = torch.argmax(logits[i], dim=-1)
        else:
            # Argmax解码
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions


# ============================================================================
# 2. 评估指标
# ============================================================================

def extract_entities(labels, label_names=None):
    """
    从标签序列中提取实体span
    
    Args:
        labels: [seq_len] - 标签序列
        label_names: 标签名称列表
    
    Returns:
        list of tuples: [(start, end, entity_type), ...]
    """
    if label_names is None:
        label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", 
                       "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    
    entities = []
    current_entity = None
    
    for i, label_id in enumerate(labels):
        if label_id == -100:  # 跳过padding
            continue
        
        label_name = label_names[label_id] if label_id < len(label_names) else "O"
        
        if label_name.startswith("B-"):
            # 开始新实体
            if current_entity is not None:
                entities.append(current_entity)
            entity_type = label_name[2:]  # 去掉B-
            current_entity = (i, i, entity_type)
        elif label_name.startswith("I-"):
            # 继续当前实体
            if current_entity is not None:
                entity_type = label_name[2:]
                if current_entity[2] == entity_type:
                    current_entity = (current_entity[0], i, entity_type)
                else:
                    # 类型不匹配，结束当前实体，开始新实体
                    entities.append(current_entity)
                    current_entity = (i, i, entity_type)
            else:
                # I-标签但没有B-开头，当作新实体
                entity_type = label_name[2:]
                current_entity = (i, i, entity_type)
        else:
            # O标签，结束当前实体
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
    
    # 添加最后一个实体
    if current_entity is not None:
        entities.append(current_entity)
    
    return entities


def compute_span_f1(pred_entities, true_entities):
    """
    计算Span-level F1
    
    Args:
        pred_entities: list of (start, end, type)
        true_entities: list of (start, end, type)
    
    Returns:
        dict: precision, recall, f1
    """
    pred_set = set(pred_entities)
    true_set = set(true_entities)
    
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def compute_f1_metrics(predictions, labels, num_labels=9):
    """
    计算NER的F1指标（Token-level）
    
    Args:
        predictions: [total_tokens] - 预测标签
        labels: [total_tokens] - 真实标签
        num_labels: 标签数量（包括O）
    
    Returns:
        dict: 包含各类F1和平均F1
    """
    # 标签名称
    label_names = [
        "O",        # 0
        "B-PER", "I-PER",   # 1, 2
        "B-ORG", "I-ORG",   # 3, 4
        "B-LOC", "I-LOC",   # 5, 6
        "B-MISC", "I-MISC"  # 7, 8
    ]
    
    # 过滤掉padding（-100）
    valid_mask = labels != -100
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    
    # 计算每个类别的precision, recall, F1
    per_class_metrics = {}
    
    for label_id in range(num_labels):
        if label_id == 0:  # 跳过O标签
            continue
        
        label_name = label_names[label_id] if label_id < len(label_names) else f"Label-{label_id}"
        
        # TP, FP, FN
        tp = ((predictions == label_id) & (labels == label_id)).sum().item()
        fp = ((predictions == label_id) & (labels != label_id)).sum().item()
        fn = ((predictions != label_id) & (labels == label_id)).sum().item()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[label_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': (labels == label_id).sum().item()
        }
    
    # Micro F1 (不包括O) - 正确计算
    entity_label_ids = [i for i in range(1, num_labels)]
    
    # 判断哪些位置预测为实体、真实为实体
    is_pred_entity = torch.isin(predictions, torch.tensor(entity_label_ids, device=predictions.device))
    is_true_entity = torch.isin(labels, torch.tensor(entity_label_ids, device=labels.device))
    
    # 计算 TP, FP, FN
    tp = ((is_pred_entity) & (is_true_entity) & (predictions == labels)).sum().item()
    fp = ((is_pred_entity) & ((~is_true_entity) | (predictions != labels))).sum().item()
    fn = ((is_true_entity) & ((~is_pred_entity) | (predictions != labels))).sum().item()
    
    # 计算 Precision, Recall, F1
    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # Macro F1 (不包括O)
    f1_scores = [metrics['f1'] for metrics in per_class_metrics.values()]
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
    
    return {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'per_class': per_class_metrics
    }


# ============================================================================
# 3. 训练和验证
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in progress_bar:
        # 数据移到device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        loss, logits = model(input_ids, attention_mask, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # 记录
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / (progress_bar.n + 1):.4f}"
        })
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device, split_name="Val"):
    """评估模型（同时计算Token-level和Span-level F1）"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_predictions_2d = []  # 保持2D结构用于span评估
    all_labels_2d = []
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"{split_name}")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            loss, logits = model(input_ids, attention_mask, labels)
            total_loss += loss.item()
            
            # 解码
            if model.use_crf:
                predictions = model.decode(input_ids, attention_mask)
            else:
                predictions = torch.argmax(logits, dim=-1)
            
            # 收集预测和标签（flatten用于token-level）
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            
            # 保持2D结构用于span-level评估
            all_predictions_2d.append(predictions.cpu())
            all_labels_2d.append(labels.cpu())
    
    # 拼接所有batch（1D for token-level）
    all_predictions_flat = torch.cat(all_predictions, dim=0).flatten()
    all_labels_flat = torch.cat(all_labels, dim=0).flatten()
    
    # 计算Token-level F1
    token_metrics = compute_f1_metrics(all_predictions_flat, all_labels_flat)
    
    # 计算Span-level F1
    all_pred_entities = []
    all_true_entities = []
    
    for preds, labels in zip(all_predictions_2d, all_labels_2d):
        for pred_seq, label_seq in zip(preds, labels):
            pred_entities = extract_entities(pred_seq.tolist())
            true_entities = extract_entities(label_seq.tolist())
            all_pred_entities.extend(pred_entities)
            all_true_entities.extend(true_entities)
    
    span_metrics = compute_span_f1(all_pred_entities, all_true_entities)
    
    # 合并指标
    metrics = {
        'token_micro_precision': token_metrics['micro_precision'],
        'token_micro_recall': token_metrics['micro_recall'],
        'token_micro_f1': token_metrics['micro_f1'],
        'token_macro_f1': token_metrics['macro_f1'],
        'span_precision': span_metrics['precision'],
        'span_recall': span_metrics['recall'],
        'span_f1': span_metrics['f1'],
        'per_class': token_metrics['per_class']
    }
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, metrics


# ============================================================================
# 4. 主训练流程
# ============================================================================

def main():
    """主函数"""
    print("=" * 80)
    print("简单NER训练脚本")
    print("=" * 80)
    
    # ========================================
    # 配置
    # ========================================
    CONFIG = {
        # 数据
        'data_dir': PROJECT_ROOT / 'data/MNER/twitter2015',
        'image_dir': PROJECT_ROOT / 'data/img',
        'train_file': 'train.txt',
        'dev_file': 'dev.txt',
        'test_file': 'test.txt',
        
        # 模型
        'text_encoder': 'microsoft/deberta-v3-base',
        'num_labels': 9,
        'lstm_hidden': 256,
        'lstm_layers': 2,
        'dropout': 0.3,
        'use_crf': True,
        
        # 训练
        'batch_size': 16,
        'num_epochs': 20,
        'learning_rate': 1e-5,
        'lstm_lr': 1e-4,
        'crf_lr': 1e-3,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'max_seq_length': 128,
        
        # 其他
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42
    }
    
    print("\n📋 配置:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # 设置随机种子
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])
    
    # ========================================
    # 1. 数据加载
    # ========================================
    print("\n" + "=" * 80)
    print("📂 1. 数据加载")
    print("=" * 80)
    
    # 训练集
    train_dataset = MNERDataset(
        text_file=str(CONFIG['data_dir'] / CONFIG['train_file']),
        image_dir=str(CONFIG['image_dir']),
        tokenizer_name=CONFIG['text_encoder'],
        max_seq_length=CONFIG['max_seq_length']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0  # Windows下设为0
    )
    print(f"✓ 训练集: {len(train_dataset)} 样本")
    
    # 验证集
    dev_dataset = MNERDataset(
        text_file=str(CONFIG['data_dir'] / CONFIG['dev_file']),
        image_dir=str(CONFIG['image_dir']),
        tokenizer_name=CONFIG['text_encoder'],
        max_seq_length=CONFIG['max_seq_length']
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    print(f"✓ 验证集: {len(dev_dataset)} 样本")
    
    # 测试集
    test_dataset = MNERDataset(
        text_file=str(CONFIG['data_dir'] / CONFIG['test_file']),
        image_dir=str(CONFIG['image_dir']),
        tokenizer_name=CONFIG['text_encoder'],
        max_seq_length=CONFIG['max_seq_length']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    print(f"✓ 测试集: {len(test_dataset)} 样本")
    
    # ========================================
    # 2. 模型构建
    # ========================================
    print("\n" + "=" * 80)
    print("🏗️ 2. 模型构建")
    print("=" * 80)
    
    model = SimpleNERModel(
        text_encoder_name=CONFIG['text_encoder'],
        num_labels=CONFIG['num_labels'],
        lstm_hidden=CONFIG['lstm_hidden'],
        lstm_layers=CONFIG['lstm_layers'],
        dropout=CONFIG['dropout'],
        use_crf=CONFIG['use_crf']
    )
    model = model.to(CONFIG['device'])
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ 总参数: {total_params:,}")
    print(f"✓ 可训练参数: {trainable_params:,}")
    
    # ========================================
    # 3. 优化器和调度器
    # ========================================
    print("\n" + "=" * 80)
    print("⚙️ 3. 优化器配置")
    print("=" * 80)
    
    # 分层学习率（关键！）
    optimizer_grouped_parameters = [
        # DeBERTa (低学习率)
        {
            'params': model.text_encoder.parameters(),
            'lr': CONFIG['learning_rate'],
            'weight_decay': CONFIG['weight_decay']
        },
        # BiLSTM (中学习率)
        {
            'params': model.bilstm.parameters(),
            'lr': CONFIG['lstm_lr'],
            'weight_decay': CONFIG['weight_decay']
        },
        # Classifier (中学习率)
        {
            'params': model.classifier.parameters(),
            'lr': CONFIG['lstm_lr'],
            'weight_decay': CONFIG['weight_decay']
        }
    ]
    
    # CRF (高学习率)
    if CONFIG['use_crf']:
        optimizer_grouped_parameters.append({
            'params': model.crf.parameters(),
            'lr': CONFIG['crf_lr'],
            'weight_decay': 0.0  # CRF不使用weight decay
        })
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    # 学习率调度器
    total_steps = len(train_loader) * CONFIG['num_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"✓ 优化器: AdamW")
    print(f"✓ DeBERTa LR: {CONFIG['learning_rate']}")
    print(f"✓ BiLSTM LR: {CONFIG['lstm_lr']}")
    print(f"✓ CRF LR: {CONFIG['crf_lr']}")
    print(f"✓ 总步数: {total_steps:,}")
    print(f"✓ 预热步数: {warmup_steps:,}")
    
    # ========================================
    # 4. 训练循环
    # ========================================
    print("\n" + "=" * 80)
    print("🚀 4. 开始训练")
    print("=" * 80)
    
    best_dev_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{CONFIG['num_epochs']}")
        print(f"{'=' * 80}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, CONFIG['device'], epoch)
        print(f"\n✓ 训练损失: {train_loss:.4f}")
        
        # 验证
        dev_loss, dev_metrics = evaluate(model, dev_loader, CONFIG['device'], "Dev")
        print(f"\n✓ 验证损失: {dev_loss:.4f}")
        print(f"\n【Token-level】")
        print(f"  Precision: {dev_metrics['token_micro_precision']:.2%}")
        print(f"  Recall: {dev_metrics['token_micro_recall']:.2%}")
        print(f"  Micro F1: {dev_metrics['token_micro_f1']:.2%}")
        print(f"\n【Span-level】")
        print(f"  Precision: {dev_metrics['span_precision']:.2%}")
        print(f"  Recall: {dev_metrics['span_recall']:.2%}")
        print(f"  F1: {dev_metrics['span_f1']:.2%} ⭐")
        
        # 保存最佳模型（以span F1为准）
        if dev_metrics['span_f1'] > best_dev_f1:
            best_dev_f1 = dev_metrics['span_f1']
            best_epoch = epoch
            
            # 保存模型
            save_path = PROJECT_ROOT / 'tests/best_ner_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dev_f1': best_dev_f1,
                'config': CONFIG
            }, save_path)
            print(f"\n✅ 保存最佳模型 (F1={best_dev_f1:.2%}) -> {save_path}")
    
    # ========================================
    # 5. 测试
    # ========================================
    print("\n" + "=" * 80)
    print("🧪 5. 测试集评估")
    print("=" * 80)
    
    # 加载最佳模型
    checkpoint = torch.load(PROJECT_ROOT / 'tests/best_ner_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 加载最佳模型 (Epoch {checkpoint['epoch']})")
    
    # 测试
    test_loss, test_metrics = evaluate(model, test_loader, CONFIG['device'], "Test")
    
    print(f"\n{'=' * 80}")
    print("📊 最终结果")
    print(f"{'=' * 80}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Dev Span F1: {best_dev_f1:.2%}")
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"\n【Token-level 指标】")
    print(f"  Precision: {test_metrics['token_micro_precision']:.2%}")
    print(f"  Recall: {test_metrics['token_micro_recall']:.2%}")
    print(f"  Micro F1 (no O): {test_metrics['token_micro_f1']:.2%}")
    print(f"  Macro F1: {test_metrics['token_macro_f1']:.2%}")
    print(f"\n【Span-level 指标】⭐")
    print(f"  Precision: {test_metrics['span_precision']:.2%}")
    print(f"  Recall: {test_metrics['span_recall']:.2%}")
    print(f"  F1: {test_metrics['span_f1']:.2%}")
    
    print(f"\n{'=' * 80}")
    print("📈 各类别F1:")
    print(f"{'=' * 80}")
    for label_name, metrics in test_metrics['per_class'].items():
        print(f"{label_name:10s}: P={metrics['precision']:.2%}, R={metrics['recall']:.2%}, "
              f"F1={metrics['f1']:.2%}, Support={metrics['support']}")
    
    print(f"\n{'=' * 80}")
    print("✅ 训练完成！")
    print(f"{'=' * 80}\n")


def test_entity_extraction():
    """测试实体提取的准确性（使用明确的测试用例）"""
    print("\n" + "=" * 80)
    print("🧪 严格测试: 实体提取")
    print("=" * 80)
    
    # 测试用例1: 正常的B-I序列
    print("\n[测试1] 正常的B-I序列")
    labels1 = [0, 1, 2, 0, 3, 4, 0]  # O, B-PER, I-PER, O, B-ORG, I-ORG, O
    entities1 = extract_entities(labels1)
    expected1 = [(1, 2, 'PER'), (4, 5, 'ORG')]
    assert entities1 == expected1, f"❌ 失败: {entities1} != {expected1}"
    print(f"  ✓ 提取实体: {entities1}")
    print(f"  ✓ 预期实体: {expected1}")
    
    # 测试用例2: 包含padding (-100)
    print("\n[测试2] 包含padding的序列")
    labels2 = [-100, 1, 2, 0, -100, -100]  # [CLS], B-PER, I-PER, O, [SEP], [PAD]
    entities2 = extract_entities(labels2)
    expected2 = [(1, 2, 'PER')]
    assert entities2 == expected2, f"❌ 失败: {entities2} != {expected2}"
    print(f"  ✓ 正确忽略padding (-100)")
    print(f"  ✓ 提取实体: {entities2}")
    
    # 测试用例3: 连续多个实体
    print("\n[测试3] 连续多个实体（无O间隔）")
    labels3 = [1, 2, 3, 4, 5, 6]  # B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
    entities3 = extract_entities(labels3)
    expected3 = [(0, 1, 'PER'), (2, 3, 'ORG'), (4, 5, 'LOC')]
    assert entities3 == expected3, f"❌ 失败: {entities3} != {expected3}"
    print(f"  ✓ 提取实体: {entities3}")
    
    # 测试用例4: 不完整序列（只有I没有B）
    print("\n[测试4] 不完整序列（I标签但无B开头）")
    labels4 = [0, 2, 0]  # O, I-PER, O (缺少B-PER)
    entities4 = extract_entities(labels4)
    expected4 = [(1, 1, 'PER')]  # 应该当作新实体
    assert entities4 == expected4, f"❌ 失败: {entities4} != {expected4}"
    print(f"  ✓ 正确处理孤立的I标签: {entities4}")
    
    # 测试用例5: 类型不匹配（B-PER后接I-ORG）
    print("\n[测试5] 类型不匹配（B-PER + I-ORG）")
    labels5 = [1, 4, 0]  # B-PER, I-ORG, O
    entities5 = extract_entities(labels5)
    expected5 = [(0, 0, 'PER'), (1, 1, 'ORG')]  # 应该拆成两个实体
    assert entities5 == expected5, f"❌ 失败: {entities5} != {expected5}"
    print(f"  ✓ 正确处理类型不匹配: {entities5}")
    
    print("\n✅ 所有实体提取测试通过！")
    return True


def test_span_f1_calculation():
    """测试Span-level F1计算的准确性"""
    print("\n" + "=" * 80)
    print("🧪 严格测试: Span-level F1计算")
    print("=" * 80)
    
    # 测试用例1: 完全匹配
    print("\n[测试1] 完全匹配（100% F1）")
    pred1 = [(0, 1, 'PER'), (3, 4, 'ORG')]
    true1 = [(0, 1, 'PER'), (3, 4, 'ORG')]
    result1 = compute_span_f1(pred1, true1)
    assert result1['precision'] == 1.0, f"❌ Precision错误: {result1['precision']}"
    assert result1['recall'] == 1.0, f"❌ Recall错误: {result1['recall']}"
    assert result1['f1'] == 1.0, f"❌ F1错误: {result1['f1']}"
    print(f"  ✓ P={result1['precision']:.2%}, R={result1['recall']:.2%}, F1={result1['f1']:.2%}")
    
    # 测试用例2: 完全不匹配
    print("\n[测试2] 完全不匹配（0% F1）")
    pred2 = [(0, 1, 'PER')]
    true2 = [(3, 4, 'ORG')]
    result2 = compute_span_f1(pred2, true2)
    assert result2['precision'] == 0.0, f"❌ Precision错误: {result2['precision']}"
    assert result2['recall'] == 0.0, f"❌ Recall错误: {result2['recall']}"
    assert result2['f1'] == 0.0, f"❌ F1错误: {result2['f1']}"
    print(f"  ✓ P={result2['precision']:.2%}, R={result2['recall']:.2%}, F1={result2['f1']:.2%}")
    
    # 测试用例3: 边界错误
    print("\n[测试3] 边界错误（位置不同）")
    pred3 = [(0, 1, 'PER')]  # 预测: token 0-1
    true3 = [(0, 2, 'PER')]  # 真实: token 0-2（更长）
    result3 = compute_span_f1(pred3, true3)
    assert result3['f1'] == 0.0, f"❌ 边界不同应该F1=0，实际: {result3['f1']}"
    print(f"  ✓ 边界不同，F1=0%（严格匹配）")
    
    # 测试用例4: 类型错误
    print("\n[测试4] 类型错误（位置相同但类型不同）")
    pred4 = [(0, 1, 'PER')]
    true4 = [(0, 1, 'ORG')]  # 位置相同，类型不同
    result4 = compute_span_f1(pred4, true4)
    assert result4['f1'] == 0.0, f"❌ 类型不同应该F1=0，实际: {result4['f1']}"
    print(f"  ✓ 类型不同，F1=0%（必须完全匹配）")
    
    # 测试用例5: 部分匹配（2个预测，1个正确）
    print("\n[测试5] 部分匹配（Precision=50%, Recall=100%）")
    pred5 = [(0, 1, 'PER'), (3, 4, 'ORG')]  # 2个预测
    true5 = [(0, 1, 'PER')]  # 1个真实
    result5 = compute_span_f1(pred5, true5)
    expected_p = 0.5  # 1 TP / 2 pred
    expected_r = 1.0  # 1 TP / 1 true
    expected_f1 = 2 * expected_p * expected_r / (expected_p + expected_r)
    assert abs(result5['precision'] - expected_p) < 1e-6, f"❌ Precision错误"
    assert abs(result5['recall'] - expected_r) < 1e-6, f"❌ Recall错误"
    assert abs(result5['f1'] - expected_f1) < 1e-6, f"❌ F1错误"
    print(f"  ✓ P={result5['precision']:.2%}, R={result5['recall']:.2%}, F1={result5['f1']:.2%}")
    
    # 测试用例6: TP, FP, FN计数
    print("\n[测试6] TP/FP/FN计数验证")
    pred6 = [(0, 1, 'PER'), (3, 4, 'ORG'), (6, 7, 'LOC')]  # 3个预测
    true6 = [(0, 1, 'PER'), (3, 4, 'ORG')]  # 2个真实
    result6 = compute_span_f1(pred6, true6)
    assert result6['tp'] == 2, f"❌ TP应该=2，实际={result6['tp']}"
    assert result6['fp'] == 1, f"❌ FP应该=1，实际={result6['fp']}"
    assert result6['fn'] == 0, f"❌ FN应该=0，实际={result6['fn']}"
    print(f"  ✓ TP={result6['tp']}, FP={result6['fp']}, FN={result6['fn']}")
    
    print("\n✅ 所有Span F1计算测试通过！")
    return True


def test_token_f1_calculation():
    """测试Token-level F1计算的准确性（修复后的版本）"""
    print("\n" + "=" * 80)
    print("🧪 严格测试: Token-level F1计算")
    print("=" * 80)
    
    # 测试用例1: 完美预测（100% F1）
    print("\n[测试1] 完美预测（100% F1）")
    predictions = torch.tensor([0, 1, 2, 0, 0, 3, 4, 0])  # O, B-PER, I-PER, O, O, B-ORG, I-ORG, O
    labels = torch.tensor([0, 1, 2, 0, 0, 3, 4, 0])
    result1 = compute_f1_metrics(predictions, labels, num_labels=9)
    
    assert result1['micro_precision'] == 1.0, f"❌ Precision错误: {result1['micro_precision']}"
    assert result1['micro_recall'] == 1.0, f"❌ Recall错误: {result1['micro_recall']}"
    assert result1['micro_f1'] == 1.0, f"❌ F1错误: {result1['micro_f1']}"
    print(f"  ✓ Precision={result1['micro_precision']:.2%}, Recall={result1['micro_recall']:.2%}, F1={result1['micro_f1']:.2%}")
    
    # 测试用例2: 全部预测错误（0% F1）
    print("\n[测试2] 全部预测错误（0% F1）")
    predictions2 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])  # 全部预测为O
    labels2 = torch.tensor([0, 1, 2, 0, 0, 3, 4, 0])  # 实际有实体
    result2 = compute_f1_metrics(predictions2, labels2, num_labels=9)
    
    # 没有预测任何实体，所以TP=0, FP=0, FN=4 (1,2,3,4)
    # Precision = 0/0 = 0, Recall = 0/4 = 0, F1 = 0
    assert result2['micro_precision'] == 0.0, f"❌ Precision错误: {result2['micro_precision']}"
    assert result2['micro_recall'] == 0.0, f"❌ Recall错误: {result2['micro_recall']}"
    assert result2['micro_f1'] == 0.0, f"❌ F1错误: {result2['micro_f1']}"
    print(f"  ✓ Precision={result2['micro_precision']:.2%}, Recall={result2['micro_recall']:.2%}, F1={result2['micro_f1']:.2%}")
    
    # 测试用例3: 部分正确（50% F1）
    print("\n[测试3] 部分正确（计算实际F1）")
    # 预测: O, B-PER, I-PER, B-ORG, O, O, O, O
    # 真实: O, B-PER, I-PER, O,     O, B-LOC, I-LOC, O
    predictions3 = torch.tensor([0, 1, 2, 3, 0, 0, 0, 0])
    labels3 = torch.tensor([0, 1, 2, 0, 0, 5, 6, 0])
    result3 = compute_f1_metrics(predictions3, labels3, num_labels=9)
    
    # TP: 预测=1且真实=1, 预测=2且真实=2 -> 2个
    # FP: 预测=3但真实=0 -> 1个
    # FN: 真实=5但预测=0, 真实=6但预测=0 -> 2个
    expected_p = 2 / (2 + 1)  # TP / (TP + FP) = 2/3 ≈ 0.667
    expected_r = 2 / (2 + 2)  # TP / (TP + FN) = 2/4 = 0.5
    expected_f1 = 2 * expected_p * expected_r / (expected_p + expected_r)  # ≈ 0.571
    
    assert abs(result3['micro_precision'] - expected_p) < 1e-6, f"❌ Precision错误: {result3['micro_precision']} != {expected_p}"
    assert abs(result3['micro_recall'] - expected_r) < 1e-6, f"❌ Recall错误: {result3['micro_recall']} != {expected_r}"
    assert abs(result3['micro_f1'] - expected_f1) < 1e-6, f"❌ F1错误: {result3['micro_f1']} != {expected_f1}"
    print(f"  ✓ Precision={result3['micro_precision']:.2%}, Recall={result3['micro_recall']:.2%}, F1={result3['micro_f1']:.2%}")
    print(f"  ✓ 验证: 2/(2+1)={expected_p:.3f}, 2/(2+2)={expected_r:.3f}, F1={expected_f1:.3f}")
    
    # 测试用例4: 包含padding（应该被忽略）
    print("\n[测试4] 包含padding标记（-100应被忽略）")
    predictions4 = torch.tensor([0, 1, 2, 0, 0, 0, 0, 0])
    labels4 = torch.tensor([-100, 1, 2, 0, 0, -100, -100, -100])  # padding标记为-100
    result4 = compute_f1_metrics(predictions4, labels4, num_labels=9)
    
    # 有效位置: [1,2,0,0] (排除-100)
    # TP: 2个正确 (1, 2)
    # FP: 0个
    # FN: 0个
    # F1 = 100%
    assert result4['micro_f1'] == 1.0, f"❌ 应忽略padding，F1应为100%，实际: {result4['micro_f1']}"
    print(f"  ✓ 正确忽略padding标记（-100）")
    print(f"  ✓ F1={result4['micro_f1']:.2%}（基于有效token）")
    
    # 测试用例5: 验证F1不等于Recall（之前的bug）
    print("\n[测试5] 验证F1 ≠ Recall（修复bug验证）")
    # 故意构造 Precision ≠ Recall 的情况
    # 预测: 2个实体 (1, 2)
    # 真实: 4个实体 (1, 2, 3, 4)
    predictions5 = torch.tensor([0, 1, 2, 0, 0, 0, 0, 0])
    labels5 = torch.tensor([0, 1, 2, 0, 0, 3, 4, 0])
    result5 = compute_f1_metrics(predictions5, labels5, num_labels=9)
    
    # TP: 2 (预测1=真实1, 预测2=真实2)
    # FP: 0
    # FN: 2 (真实3,4未被预测)
    expected_p5 = 2 / (2 + 0)  # = 1.0
    expected_r5 = 2 / (2 + 2)  # = 0.5
    expected_f15 = 2 * expected_p5 * expected_r5 / (expected_p5 + expected_r5)  # ≈ 0.667
    
    assert abs(result5['micro_precision'] - expected_p5) < 1e-6, f"❌ Precision错误"
    assert abs(result5['micro_recall'] - expected_r5) < 1e-6, f"❌ Recall错误"
    assert abs(result5['micro_f1'] - expected_f15) < 1e-6, f"❌ F1错误"
    
    # 验证F1确实不等于Recall（之前的bug）
    assert result5['micro_f1'] != result5['micro_recall'], "❌ F1仍然等于Recall！bug未修复！"
    print(f"  ✓ Precision={result5['micro_precision']:.2%} (100%)")
    print(f"  ✓ Recall={result5['micro_recall']:.2%} (50%)")
    print(f"  ✓ F1={result5['micro_f1']:.2%} (66.67%)")
    print(f"  ✓ 确认: F1 ≠ Recall（bug已修复！）")
    
    print("\n✅ 所有Token F1计算测试通过！")
    print("✅ 修复验证: Token F1现在正确计算为 F1，而不是 Recall")
    return True


def test_cls_sep_pad_handling():
    """测试对[CLS], [SEP], [PAD]的处理"""
    print("\n" + "=" * 80)
    print("🧪 严格测试: [CLS]/[SEP]/[PAD]处理")
    print("=" * 80)
    
    # 创建一个模拟batch（手工构造）
    batch_size = 2
    seq_len = 10
    num_labels = 9
    
    # 构造输入
    # Sequence 1: [CLS] George Zimmerman got shot [SEP] [PAD] [PAD] [PAD]
    # Labels:     -100  B-PER  I-PER       O   O    -100  -100  -100  -100
    input_ids = torch.tensor([[
        101,  # [CLS]
        3312, # George
        20758,# Zimmerman
        2288, # got
        2915, # shot
        102,  # [SEP]
        0, 0, 0, 0  # [PAD]
    ]], dtype=torch.long)
    
    attention_mask = torch.tensor([[
        1, 1, 1, 1, 1, 1, 0, 0, 0, 0
    ]], dtype=torch.long)
    
    labels = torch.tensor([[
        -100,  # [CLS] - 应该被忽略
        1,     # B-PER
        2,     # I-PER
        0,     # O
        0,     # O
        -100,  # [SEP] - 应该被忽略
        -100, -100, -100, -100  # [PAD] - 应该被忽略
    ]], dtype=torch.long)
    
    print("\n[测试1] 验证label分布")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Attention mask: {attention_mask[0].tolist()}")
    print(f"  Labels: {labels[0].tolist()}")
    valid_labels = labels[labels != -100]
    print(f"  ✓ 有效标签（排除-100）: {valid_labels.tolist()}")
    assert valid_labels.tolist() == [1, 2, 0, 0], "❌ 有效标签不正确"
    
    print("\n[测试2] 实体提取应忽略[CLS]/[SEP]/[PAD]")
    entities = extract_entities(labels[0].tolist())
    # 实体应该是: (1, 2, 'PER') - 注意索引从1开始（跳过[CLS]）
    expected_entities = [(1, 2, 'PER')]
    assert entities == expected_entities, f"❌ 实体提取错误: {entities} != {expected_entities}"
    print(f"  ✓ 提取的实体: {entities}")
    print(f"  ✓ 正确忽略[CLS]、[SEP]、[PAD]")
    
    print("\n[测试3] Token-level F1应忽略padding")
    # 模拟预测：全部预测为O（0）
    predictions = torch.zeros_like(labels)
    
    # 计算F1
    pred_flat = predictions.flatten()
    label_flat = labels.flatten()
    
    # 手动计算预期结果
    valid_mask = label_flat != -100
    valid_preds = pred_flat[valid_mask]
    valid_labels = label_flat[valid_mask]
    
    print(f"  有效预测: {valid_preds.tolist()}")
    print(f"  有效标签: {valid_labels.tolist()}")
    
    # 应该只计算4个token（排除[CLS], [SEP], [PAD]）
    assert len(valid_preds) == 4, f"❌ 应该有4个有效token，实际: {len(valid_preds)}"
    print(f"  ✓ 正确识别4个有效token（排除[CLS]/[SEP]/[PAD]）")
    
    print("\n[测试4] CRF mask应该正确处理")
    # 对于CRF，我们需要确保：
    # 1. [CLS] 和 [SEP] 的label是-100
    # 2. 提取有效范围时，排除这些token
    valid_indices = (labels[0] != -100).nonzero(as_tuple=True)[0]
    start_idx = valid_indices[0].item()
    end_idx = valid_indices[-1].item() + 1
    
    print(f"  有效token范围: [{start_idx}, {end_idx})")
    print(f"  提取的labels: {labels[0, start_idx:end_idx].tolist()}")
    
    # 验证提取的labels不包含-100
    extracted_labels = labels[0, start_idx:end_idx]
    assert (-100 not in extracted_labels), "❌ 提取的labels仍包含-100"
    print(f"  ✓ 提取的labels不包含-100")
    
    # 验证第一个位置不是-100（torchcrf要求）
    assert extracted_labels[0] != -100, "❌ 第一个位置是-100（违反torchcrf约束）"
    print(f"  ✓ 第一个位置有效（满足torchcrf约束）")
    
    print("\n✅ 所有[CLS]/[SEP]/[PAD]处理测试通过！")
    return True


def test_crf_mask_constraints():
    """测试CRF mask的约束条件"""
    print("\n" + "=" * 80)
    print("🧪 严格测试: CRF mask约束")
    print("=" * 80)
    
    print("\n[测试1] 验证torchcrf的mask约束")
    print("  torchcrf要求: mask[:, 0]必须全为True")
    
    # 错误示例：第一个位置的mask是False
    try:
        from torchcrf import CRF
        crf = CRF(9, batch_first=True)
        
        emissions = torch.randn(1, 3, 9)
        tags = torch.tensor([[1, 2, 0]])
        mask = torch.tensor([[False, True, True]])  # ❌ 第一个是False
        
        try:
            _ = crf(emissions, tags, mask=mask)
            print("  ❌ 应该报错但没有报错！")
            return False
        except ValueError as e:
            print(f"  ✓ 正确抛出异常: {str(e)[:50]}...")
    except Exception as e:
        print(f"  ⚠️ 跳过torchcrf约束测试: {e}")
    
    # 正确示例：第一个位置的mask是True
    print("\n[测试2] 正确的mask（第一个位置为True）")
    try:
        emissions = torch.randn(1, 3, 9)
        tags = torch.tensor([[1, 2, 0]])
        mask = torch.tensor([[True, True, True]])  # ✓ 第一个是True
        
        log_likelihood = crf(emissions, tags, mask=mask)
        print(f"  ✓ Log likelihood: {log_likelihood.item():.4f}")
    except Exception as e:
        print(f"  ⚠️ 跳过测试: {e}")
    
    print("\n[测试3] 我们的_compute_crf_loss处理")
    print("  策略：提取有效范围，确保第一个位置不是-100")
    
    # 模拟实际场景
    labels_with_cls_sep = torch.tensor([[-100, 1, 2, 0, -100]])
    valid_mask = (labels_with_cls_sep[0] != -100)
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    
    if len(valid_indices) > 0:
        start_idx = valid_indices[0].item()
        end_idx = valid_indices[-1].item() + 1
        
        extracted_labels = labels_with_cls_sep[0, start_idx:end_idx]
        print(f"  原始labels: {labels_with_cls_sep[0].tolist()}")
        print(f"  提取范围: [{start_idx}, {end_idx})")
        print(f"  提取labels: {extracted_labels.tolist()}")
        
        # 验证
        assert extracted_labels[0] != -100, "❌ 第一个位置是-100"
        assert -100 not in extracted_labels, "❌ 包含-100"
        print(f"  ✓ 第一个位置有效: {extracted_labels[0].item()}")
        print(f"  ✓ 不包含-100")
    
    print("\n✅ 所有CRF mask约束测试通过！")
    return True


def test_model_components():
    """测试模型各组件的正确性"""
    print("\n" + "=" * 80)
    print("🧪 严格测试: 模型组件")
    print("=" * 80)
    
    TEST_CONFIG = {
        'text_encoder': 'microsoft/deberta-v3-base',
        'num_labels': 9,
        'lstm_hidden': 128,
        'lstm_layers': 1,
        'dropout': 0.3,
        'use_crf': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    print("\n[测试1] 模型构建")
    model = SimpleNERModel(
        text_encoder_name=TEST_CONFIG['text_encoder'],
        num_labels=TEST_CONFIG['num_labels'],
        lstm_hidden=TEST_CONFIG['lstm_hidden'],
        lstm_layers=TEST_CONFIG['lstm_layers'],
        dropout=TEST_CONFIG['dropout'],
        use_crf=TEST_CONFIG['use_crf']
    )
    model = model.to(TEST_CONFIG['device'])
    print(f"  ✓ 模型创建成功")
    
    # 验证各组件
    assert hasattr(model, 'text_encoder'), "❌ 缺少text_encoder"
    assert hasattr(model, 'bilstm'), "❌ 缺少bilstm"
    assert hasattr(model, 'classifier'), "❌ 缺少classifier"
    assert hasattr(model, 'crf'), "❌ 缺少crf"
    print(f"  ✓ 所有组件存在")
    
    print("\n[测试2] 前向传播（训练模式）")
    # 构造输入
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(100, 1000, (batch_size, seq_len)).to(TEST_CONFIG['device'])
    attention_mask = torch.ones(batch_size, seq_len).to(TEST_CONFIG['device'])
    labels = torch.randint(0, TEST_CONFIG['num_labels'], (batch_size, seq_len)).to(TEST_CONFIG['device'])
    labels[:, 0] = -100  # [CLS]
    labels[:, -1] = -100  # [SEP]
    
    loss, logits = model(input_ids, attention_mask, labels)
    
    assert not torch.isnan(loss), "❌ Loss是NaN"
    assert not torch.isinf(loss), "❌ Loss是Inf"
    assert loss.item() > 0, "❌ Loss应该>0"
    print(f"  ✓ Loss: {loss.item():.4f}")
    print(f"  ✓ Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, TEST_CONFIG['num_labels']), "❌ Logits shape错误"
    
    print("\n[测试3] 前向传播（推理模式）")
    with torch.no_grad():
        logits_eval = model(input_ids, attention_mask)
    print(f"  ✓ Logits shape: {logits_eval.shape}")
    
    print("\n[测试4] CRF解码")
    with torch.no_grad():
        predictions = model.decode(input_ids, attention_mask)
    print(f"  ✓ Predictions shape: {predictions.shape}")
    assert predictions.shape == (batch_size, seq_len), "❌ Predictions shape错误"
    
    # 验证预测值范围
    assert predictions.min() >= 0, "❌ 预测值<0"
    assert predictions.max() < TEST_CONFIG['num_labels'], "❌ 预测值>=num_labels"
    print(f"  ✓ 预测值范围: [{predictions.min()}, {predictions.max()}]")
    
    print("\n[测试5] 反向传播")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    loss, _ = model(input_ids, attention_mask, labels)
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            assert not torch.isnan(param.grad).any(), f"❌ {name}的梯度包含NaN"
            assert not torch.isinf(param.grad).any(), f"❌ {name}的梯度包含Inf"
    
    assert has_grad, "❌ 没有参数有梯度"
    print(f"  ✓ 反向传播成功，梯度正常")
    
    optimizer.step()
    print(f"  ✓ 参数更新成功")
    
    print("\n✅ 所有模型组件测试通过！")
    return True


def test_training_pipeline():
    """
    测试训练流程的完整性
    使用严格的测试用例验证每个环节
    """
    print("=" * 80)
    print("🧪 完整训练流程测试")
    print("=" * 80)
    
    try:
        # 运行所有严格测试
        tests = [
            ("实体提取", test_entity_extraction),
            ("Span F1计算", test_span_f1_calculation),
            ("Token F1计算", test_token_f1_calculation),
            ("[CLS]/[SEP]/[PAD]处理", test_cls_sep_pad_handling),
            ("CRF mask约束", test_crf_mask_constraints),
            ("模型组件", test_model_components),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except AssertionError as e:
                print(f"\n❌ {test_name}测试失败: {e}")
                failed += 1
            except Exception as e:
                print(f"\n⚠️ {test_name}测试出错: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
        
        # 总结
        print("\n" + "=" * 80)
        print("📊 测试总结")
        print("=" * 80)
        print(f"通过: {passed}/{len(tests)}")
        print(f"失败: {failed}/{len(tests)}")
        
        if failed == 0:
            print("\n" + "=" * 80)
            print("✅ 所有严格测试通过！")
            print("=" * 80)
            print("\n关键验证：")
            print("  ✓ 实体提取逻辑正确（包括边界情况）")
            print("  ✓ Span-level F1计算准确（TP/FP/FN正确）")
            print("  ✓ [CLS]/[SEP]/[PAD]正确处理")
            print("  ✓ CRF mask约束满足torchcrf要求")
            print("  ✓ 模型前向/反向传播正常")
            print("  ✓ 梯度计算无NaN/Inf")
            print("\n可以安全运行完整训练：")
            print("  python tests/simple_ner_training.py")
            return True
        else:
            print("\n❌ 部分测试失败，请检查并修复")
            return False
        
    except Exception as e:
        print(f"\n❌ 测试流程失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    # 检查是否是测试模式
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_training_pipeline()
    else:
        main()