# models/task_heads/mate_head.py
import torch
import torch.nn as nn

class MATEHead(nn.Module):
    def __init__(self, input_dim, num_labels, dropout_prob=0.1, hidden_dim=None):
        """
        :param input_dim: 输入特征维度（来自多模态融合后的输出维度）
        :param num_labels: 标签数（对于 MATE 任务通常为 3：O、B、I）
        :param dropout_prob: dropout 概率
        :param hidden_dim: 隐藏层维度，默认为 input_dim
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, seq_features):
        """
        :param seq_features: shape (batch_size, seq_len, input_dim)
        :return: logits, shape (batch_size, seq_len, num_labels)
        """
        x = self.dropout(seq_features)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
