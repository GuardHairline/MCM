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
        # 共享部分：不依赖于 num_labels
        self.shared_layer = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        # 任务特定的输出层：仅依赖于任务的标签数
        self.classifier = nn.Linear(hidden_dim, num_labels)  # 分类器输出7个标签

    def forward(self, seq_features):
        """
        :param seq_features: shape (batch_size, seq_len, input_dim)
        :return: logits, shape (batch_size, seq_len, num_labels)
        """
        shared_out = self.shared_layer(seq_features)
        logits = self.classifier(shared_out)
        return logits
