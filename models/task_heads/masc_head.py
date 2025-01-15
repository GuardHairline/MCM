# models/task_heads/masc_head.py
import torch
import torch.nn as nn

class MASCHead(nn.Module):
    def __init__(self, input_dim, num_labels=3):
        """
        :param input_dim: 多模态融合后特征的维度
        :param num_labels: 情感极性标签数 (-1, 0, 1) -> 3
        """
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, fused_feat):
        """
        :param fused_feat: [batch_size, input_dim]
        :return: logits [batch_size, num_labels]
        """
        logits = self.classifier(fused_feat)
        return logits
