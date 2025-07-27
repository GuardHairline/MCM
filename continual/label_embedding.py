# continual/label_embedding.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json
from transformers import AutoTokenizer, AutoModel

class GlobalLabelEmbedding(nn.Module):
    """
    支持跨任务共享的标签嵌入，包含标签分组/分层功能：
    1. 每个 (task, label_id) 分配一个 idx；
    2. 维护可训练 Embedding；
    3. 支持标签语义分组和相似度正则化；
    4. 可导出 json 方便推理时加载。
    5. 支持外部embedding初始化（如deberta-v3-base生成的label embedding）
    """
    def __init__(self, label2idx: dict, emb_dim: int = 768, 
                 label_groups: Optional[Dict[str, List[Tuple[str, int]]]] = None,
                 use_similarity_regularization: bool = True,
                 similarity_weight: float = 0.1,
                 pretrained_embeddings: Optional[Dict[Tuple[str, int], torch.Tensor]] = None):
        super().__init__()
        self.label2idx = label2idx          # {(task, label_id): global_id}
        self.idx2label = {v: k for k, v in label2idx.items()}
        self.embedding = nn.Embedding(len(label2idx), emb_dim)
        self.emb_dim = emb_dim
        self.label_groups = label_groups or {}
        self.use_similarity_regularization = use_similarity_regularization
        self.similarity_weight = similarity_weight

        if pretrained_embeddings is not None:
            with torch.no_grad():
                for key, idx in self.label2idx.items():
                    if key in pretrained_embeddings:
                        self.embedding.weight[idx] = pretrained_embeddings[key][:emb_dim]
        else:
            self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """根据标签分组初始化嵌入"""
        if not self.label_groups:
            return
        with torch.no_grad():
            for group_name, labels in self.label_groups.items():
                if not labels:
                    continue
                # 为每个组生成一个基础向量
                base_vector = torch.randn(self.emb_dim) * 0.1
                # 为组内每个标签分配相近的嵌入
                for task_name, label_id in labels:
                    if (task_name, label_id) in self.label2idx:
                        global_idx = self.label2idx[(task_name, label_id)]
                        perturbed_vector = base_vector + torch.randn(self.emb_dim) * 0.01
                        self.embedding.weight[global_idx] = perturbed_vector

    def get_similarity_loss(self) -> torch.Tensor:
        """计算标签相似度正则化损失"""
        if not self.use_similarity_regularization or not self.label_groups:
            return torch.tensor(0.0, device=self.embedding.weight.device)
        
        total_loss = 0.0
        count = 0
        
        for group_name, labels in self.label_groups.items():
            if len(labels) < 2:
                continue
                
            # 获取组内所有标签的嵌入
            group_embeddings = []
            for task_name, label_id in labels:
                if (task_name, label_id) in self.label2idx:
                    global_idx = self.label2idx[(task_name, label_id)]
                    group_embeddings.append(self.embedding.weight[global_idx])
            
            if len(group_embeddings) < 2:
                continue
                
            group_embeddings = torch.stack(group_embeddings)
            
            # 计算组内标签的余弦相似度损失（让它们更相似）
            similarities = F.cosine_similarity(
                group_embeddings.unsqueeze(1), 
                group_embeddings.unsqueeze(0), 
                dim=2
            )
            
            # 对角线元素应该为1，其他元素应该接近1
            mask = torch.ones_like(similarities) - torch.eye(len(group_embeddings), device=similarities.device)
            loss = torch.mean((similarities - 1.0) ** 2 * mask)
            total_loss += loss
            count += 1
        
        return total_loss / max(count, 1) * self.similarity_weight

    def forward(self, task_name: str, label_ids: torch.Tensor):
        """
        label_ids: tensor([...])  本任务局部 id
        返回对应的 global label 向量 (…, emb_dim)
        """
        flat = label_ids.view(-1)
        global_idx = torch.tensor(
            [self.label2idx[(task_name, int(l))] for l in flat],
            device=label_ids.device
        )
        emb = self.embedding(global_idx).view(*label_ids.shape, -1)
        return emb
    
    def get_all_label_embeddings(self, task_name: str) -> torch.Tensor:
        """获取指定任务的所有标签嵌入"""
        task_labels = [(t, l) for (t, l) in self.label2idx.keys() if t == task_name]
        if not task_labels:
            return torch.empty(0, self.emb_dim, device=self.embedding.weight.device)
        
        global_indices = [self.label2idx[label] for label in task_labels]
        return self.embedding.weight[global_indices]

    def export(self, path):  # 推理用
        torch.save({
            'state': self.state_dict(), 
            'map': self.label2idx,
            'groups': self.label_groups,
            'emb_dim': self.emb_dim
        }, path)

    @staticmethod
    def load(path, device='cpu'):
        ckpt = torch.load(path, map_location=device)
        obj = GlobalLabelEmbedding(
            ckpt['map'], 
            emb_dim=ckpt.get('emb_dim', 768),
            label_groups=ckpt.get('groups', {})
        )
        obj.load_state_dict(ckpt['state'])
        return obj

    def get_task_num_labels(self, task):
        """获取指定任务的标签数量"""
        # 从 label2idx 中统计指定任务的标签数量
        task_labels = [label_id for (task_name, label_id) in self.label2idx.keys() if task_name == task]
        return len(task_labels)
    def freeze_seen_labels(self, new_task_name:str, new_num_labels:int):
        """
        训练新任务前调用。把旧任务的 label embedding 的 grad 置 0，
        仅允许新任务标签更新。
        """
        new_indices = [self.label2idx[(new_task_name, i)] for i in range(new_num_labels)]
        def hook(grad):
            mask = torch.zeros_like(grad)
            mask[new_indices] = 1
            return grad * mask
        # 先移除旧 hook，避免重复挂
        if hasattr(self, "_grad_handle"):
            self._grad_handle.remove()
        self._grad_handle = self.embedding.weight.register_hook(hook)

def create_label_groups() -> Dict[str, List[Tuple[str, int]]]:
    """
    根据你的任务标签创建语义分组
    """
    return {
        "O": [  # 非目标标签
            ("mabsa", 0), ("mate", 0), ("mner", 0)
        ],
        "NEG": [  # 负向情感/否定
            ("mabsa", 1), ("mabsa", 2),  # B-NEG, I-NEG
            ("masc", 0)   # NEG (映射后)
        ],
        "NEU": [  # 中性情感
            ("mabsa", 3), ("mabsa", 4),  # B-NEU, I-NEU
            ("masc", 1)   # NEU (映射后)
        ],
        "POS": [  # 正向情感/肯定
            ("mabsa", 5), ("mabsa", 6),  # B-POS, I-POS
            ("masc", 2)   # POS (映射后)
        ],
        "B_ENTITY": [  # 实体开始标签
            ("mate", 1),      # B
            ("mner", 1), ("mner", 3), ("mner", 5), ("mner", 7),  # B-PER, B-ORG, B-LOC, B-MISC
        ],
        "I_ENTITY": [  # 实体内部标签
            ("mate", 2),      # I
            ("mner", 2), ("mner", 4), ("mner", 6), ("mner", 8),  # I-PER, I-ORG, I-LOC, I-MISC
        ]
    }

def build_global_label_mapping() -> Dict[Tuple[str, int], int]:
    task_labels = {
        "mabsa": [0, 1, 2, 3, 4, 5, 6],
        "masc": [0, 1, 2],
        "mate": [0, 1, 2],
        "mner": [0, 1, 2, 3, 4, 5, 6, 7, 8]
    }
    global_idx = 0
    label2idx = {}
    for task_name, labels in task_labels.items():
        for label_id in labels:
            label2idx[(task_name, label_id)] = global_idx
            global_idx += 1
    return label2idx

def get_label_text_mapping() -> Dict[Tuple[str, int], str]:
    """
    返回每个 (task, label_id) 的标签语义文本
    """
    return {
        ("mabsa", 0): "outside",
        ("mabsa", 1): "begin negative", ("mabsa", 2): "inside negative",
        ("mabsa", 3): "begin neutral", ("mabsa", 4): "inside neutral",
        ("mabsa", 5): "begin positive", ("mabsa", 6): "inside positive",
        ("masc", 0): "negative", ("masc", 1): "neutral", ("masc", 2): "positive",
        ("mate", 0): "outside", ("mate", 1): "begin entity", ("mate", 2): "inside entity",
        ("mner", 0): "outside",
        ("mner", 1): "begin person", ("mner", 2): "inside person",
        ("mner", 3): "begin organization", ("mner", 4): "inside organization",
        ("mner", 5): "begin location", ("mner", 6): "inside location",
        ("mner", 7): "begin miscellaneous", ("mner", 8): "inside miscellaneous",
    }

def generate_label_embeddings(label_texts, emb_dim=768, device="cpu"):
    """
    用deberta-v3-base生成所有标签的embedding，返回{(task, label_id): tensor}
    """
    tokenizer = AutoTokenizer.from_pretrained("downloaded_model/deberta-v3-base", use_fast=False)
    model = AutoModel.from_pretrained("downloaded_model/deberta-v3-base").to(device)
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for key, text in label_texts.items():
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]  # 取CLS
            embeddings[key] = emb.squeeze(0).cpu()
    return embeddings