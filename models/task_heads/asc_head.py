# models/task_heads/asc_head.py
import torch
import torch.nn as nn

class ASCHead(nn.Module):
    def __init__(self, input_dim, num_labels, dropout_prob=0.1, hidden_dim=None):
        """
        :param input_dim: 输入特征维度（来自多模态融合后的输出维度）
        :param num_labels: 标签数
        :param dropout_prob:
        :param hidden_dim:
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        # 共享部分：不依赖于 num_labels
        self.shared_layer = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        # 任务特定的输出层：仅依赖于任务的标签数
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, cls_features):  # (b, input_dim)
        shared_out = self.shared_layer(cls_features)
        logits = self.classifier(shared_out)
        return logits