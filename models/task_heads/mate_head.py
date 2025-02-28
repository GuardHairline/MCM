# models/task_heads/mate_head.py
import torch
import torch.nn as nn
from TorchCRF import CRF

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
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, seq_features, labels=None, mask=None):
        """
        :param seq_features: Tensor, shape (batch_size, seq_len, input_dim)
        :param labels: Tensor, shape (batch_size, seq_len)，取值范围根据 num_labels（mate: 0,1,2；mner: 0~8），
                       对于特殊 token 应为 -100
        :param mask: ByteTensor or BoolTensor, shape (batch_size, seq_len)，标记哪些位置是有效的。
                     如果为 None，则根据 labels != -100 自动构造。
        :return: 如果 labels 不为 None，则返回训练时的 loss；否则返回预测的标签序列（List[List[int]]）
        """
        x = self.dropout(seq_features)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        emissions = self.classifier(x)  # shape: (batch_size, seq_len, num_labels)

        if mask is None:
            # 构造 mask：有效 token 的位置为 True
            mask = labels != -100
        mask[:, 0] = True

        if labels is not None:
            # CRF 的 log_likelihood 返回的是对数似然，训练目标为负对数似然
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            # 解码预测标签
            predicted_labels = self.crf.decode(emissions, mask=mask)
            return predicted_labels
