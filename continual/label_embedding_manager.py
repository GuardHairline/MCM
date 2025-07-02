# continual/label_embedding_manager.py
import os
import torch
from typing import Optional, Dict, Any
from .label_embedding import GlobalLabelEmbedding, create_label_groups, build_global_label_mapping

class LabelEmbeddingManager:
    """
    标签嵌入管理器，负责全局标签嵌入的创建、加载、保存和管理
    """
    
    def __init__(self, emb_dim: int = 128, 
                 use_similarity_regularization: bool = True,
                 similarity_weight: float = 0.1):
        self.emb_dim = emb_dim
        self.use_similarity_regularization = use_similarity_regularization
        self.similarity_weight = similarity_weight
        self.label_embedding: Optional[GlobalLabelEmbedding] = None
        
    def create_or_load_embedding(self, 
                                embedding_path: Optional[str] = None,
                                device: str = 'cpu') -> GlobalLabelEmbedding:
        """
        创建新的标签嵌入或从文件加载
        """
        if embedding_path and os.path.exists(embedding_path):
            print(f"Loading label embedding from {embedding_path}")
            self.label_embedding = GlobalLabelEmbedding.load(embedding_path, device)
        else:
            print("Creating new label embedding")
            label2idx = build_global_label_mapping()
            label_groups = create_label_groups()
            
            self.label_embedding = GlobalLabelEmbedding(
                label2idx=label2idx,
                emb_dim=self.emb_dim,
                label_groups=label_groups,
                use_similarity_regularization=self.use_similarity_regularization,
                similarity_weight=self.similarity_weight
            ).to(device)
            
            if embedding_path:
                self.save_embedding(embedding_path)
        
        return self.label_embedding
    
    def save_embedding(self, path: str):
        """保存标签嵌入"""
        if self.label_embedding is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.label_embedding.export(path)
            print(f"Label embedding saved to {path}")
    
    def get_embedding(self) -> Optional[GlobalLabelEmbedding]:
        """获取当前标签嵌入"""
        return self.label_embedding
    
    def get_similarity_loss(self) -> torch.Tensor:
        """获取相似度正则化损失"""
        if self.label_embedding is not None:
            return self.label_embedding.get_similarity_loss()
        return torch.tensor(0.0)
    
    def get_task_labels(self, task_name: str) -> Dict[int, int]:
        """获取指定任务的标签映射"""
        if self.label_embedding is None:
            return {}
        
        task_labels = {}
        for (task, label_id), global_idx in self.label_embedding.label2idx.items():
            if task == task_name:
                task_labels[label_id] = global_idx
        
        return task_labels
    
    def get_task_num_labels(self, task_name: str) -> int:
        """获取指定任务的标签数量"""
        return len(self.get_task_labels(task_name))
    
    def print_label_mapping(self):
        """打印标签映射信息"""
        if self.label_embedding is None:
            print("No label embedding loaded")
            return
        
        print("Global Label Mapping:")
        for (task, label_id), global_idx in self.label_embedding.label2idx.items():
            print(f"  ({task}, {label_id}) -> {global_idx}")
        
        print("\nLabel Groups:")
        for group_name, labels in self.label_embedding.label_groups.items():
            print(f"  {group_name}: {labels}") 