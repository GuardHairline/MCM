# models/task_heads/mate_head.py
import torch
import torch.nn as nn

class MATEHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, seq_features):  # (b, seq_len, input_dim)
        return self.classifier(seq_features)  # => (b, seq_len, num_labels)
