import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from transformers import CLIPProcessor

def create_clap4clip_model(config: Dict) -> nn.Module:
    """
    创建CLAP4CLIP模型
    Args:
        config: 配置字典
    Returns:
        model: CLAP4CLIP模型
    """
    from .clap4clip import CLAP4CLIP
    
    model = CLAP4CLIP(
        text_model_name=config.get('text_model_name', 'openai/clip-vit-base-patch32'),
        image_model_name=config.get('image_model_name', 'openai/clip-vit-base-patch32'),
        num_labels=config.get('num_labels', 3),
        dropout_prob=config.get('dropout_prob', 0.1),
        adapter_size=config.get('adapter_size', 64),
        finetune_lambda=config.get('finetune_lambda', 0.1),
        temperature=config.get('temperature', 0.07)
    )
    
    return model

def compute_clap4clip_loss(logits: torch.Tensor, 
                          labels: torch.Tensor,
                          text_features: torch.Tensor,
                          image_features: torch.Tensor,
                          model: nn.Module,
                          contrastive_weight: float = 0.1) -> torch.Tensor:
    """
    计算CLAP4CLIP的总损失
    Args:
        logits: 分类logits
        labels: 真实标签
        text_features: 文本特征
        image_features: 图像特征
        model: CLAP4CLIP模型
        contrastive_weight: 对比学习损失权重
    Returns:
        total_loss: 总损失
    """
    # 分类损失
    classification_loss = F.cross_entropy(logits, labels)
    
    # 对比学习损失
    contrastive_loss = model.compute_contrastive_loss(text_features, image_features)
    
    # 总损失
    total_loss = classification_loss + contrastive_weight * contrastive_loss
    
    return total_loss

def get_clip_processor(model_name: str = 'openai/clip-vit-base-patch32') -> CLIPProcessor:
    """
    获取CLIP处理器
    Args:
        model_name: CLIP模型名称
    Returns:
        processor: CLIP处理器
    """
    return CLIPProcessor.from_pretrained(model_name)

def preprocess_clap4clip_data(texts: List[str], 
                             images: List, 
                             processor: CLIPProcessor,
                             max_length: int = 77) -> Dict[str, torch.Tensor]:
    """
    预处理CLAP4CLIP数据
    Args:
        texts: 文本列表
        images: 图像列表
        processor: CLIP处理器
        max_length: 最大文本长度
    Returns:
        processed_data: 处理后的数据
    """
    # 处理文本
    text_inputs = processor(
        text=texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 处理图像
    image_inputs = processor(
        images=images,
        return_tensors="pt"
    )
    
    return {
        'input_ids': text_inputs['input_ids'],
        'attention_mask': text_inputs['attention_mask'],
        'image_tensor': image_inputs['pixel_values']
    }

def evaluate_clap4clip_performance(model: nn.Module,
                                 test_loader,
                                 device: str,
                                 task_name: str) -> Dict[str, float]:
    """
    评估CLAP4CLIP性能
    Args:
        model: CLAP4CLIP模型
        test_loader: 测试数据加载器
        device: 设备
        task_name: 任务名称
    Returns:
        metrics: 评估指标
    """
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_tensor = batch['image_tensor'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_tensor=image_tensor,
                task_name=task_name
            )
            
            # 计算损失
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            
            # 收集预测结果
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    avg_loss = total_loss / len(test_loader)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': avg_loss
    }

def save_clap4clip_checkpoint(model: nn.Module,
                             optimizer,
                             epoch: int,
                             save_path: str,
                             task_name: str):
    """
    保存CLAP4CLIP检查点
    Args:
        model: CLAP4CLIP模型
        optimizer: 优化器
        epoch: 当前轮次
        save_path: 保存路径
        task_name: 任务名称
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'task_name': task_name,
        'task_heads': model.task_heads,
        'current_task': model.current_task
    }
    
    torch.save(checkpoint, save_path)

def load_clap4clip_checkpoint(model: nn.Module,
                             checkpoint_path: str,
                             device: str):
    """
    加载CLAP4CLIP检查点
    Args:
        model: CLAP4CLIP模型
        checkpoint_path: 检查点路径
        device: 设备
    Returns:
        optimizer: 优化器
        epoch: 轮次
        task_name: 任务名称
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.task_heads = checkpoint.get('task_heads', {})
    model.current_task = checkpoint.get('current_task', None)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return optimizer, checkpoint['epoch'], checkpoint['task_name']

def compute_task_similarity(model: nn.Module,
                          task1_features: torch.Tensor,
                          task2_features: torch.Tensor) -> float:
    """
    计算任务相似度
    Args:
        model: CLAP4CLIP模型
        task1_features: 任务1特征
        task2_features: 任务2特征
    Returns:
        similarity: 相似度分数
    """
    # 归一化特征
    task1_norm = F.normalize(task1_features, dim=-1)
    task2_norm = F.normalize(task2_features, dim=-1)
    
    # 计算余弦相似度
    similarity = F.cosine_similarity(task1_norm, task2_norm, dim=-1)
    
    return similarity.mean().item()

def adaptive_learning_rate(epoch: int, 
                          base_lr: float = 1e-5,
                          warmup_epochs: int = 5,
                          decay_factor: float = 0.9) -> float:
    """
    自适应学习率调度
    Args:
        epoch: 当前轮次
        base_lr: 基础学习率
        warmup_epochs: 预热轮次
        decay_factor: 衰减因子
    Returns:
        lr: 当前学习率
    """
    if epoch < warmup_epochs:
        # 预热阶段
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        # 衰减阶段
        lr = base_lr * (decay_factor ** (epoch - warmup_epochs))
    
    return lr
