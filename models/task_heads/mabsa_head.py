import torch
import torch.nn as nn

class MABSAHead(nn.Module):
    def __init__(self, input_dim, num_labels=7, dropout_prob=0.1, hidden_dim=None):
        """
        :param input_dim: 输入特征维度（来自多模态融合后的输出维度）
        :param num_labels: 标签数（7个标签：O, B-positive, B-neutral, B-negative, I-positive, I-neutral, I-negative）
        :param dropout_prob: dropout 概率
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(input_dim, hidden_dim)  # 增加一个线性层来调整特征维度
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_labels)  # 分类器输出7个标签

    def forward(self, seq_features):
        """
        :param seq_features: shape (batch_size, seq_len, input_dim)
        :return: logits, shape (batch_size, seq_len, num_labels)
        """
        x = self.dropout(seq_features)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)  # 生成7个类别的logits
        return logits
