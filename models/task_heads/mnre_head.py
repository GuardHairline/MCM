# models/task_heads/re_head.py
import torch
import torch.nn as nn

class MNREHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, cls_features):  # (b, input_dim)
        return self.classifier(cls_features)  # => (b, num_relations)
