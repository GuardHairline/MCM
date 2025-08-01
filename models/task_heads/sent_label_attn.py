# \models\task_heads\label_attention_sent_head.py
import torch
import torch.nn as nn
from continual.label_embedding import GlobalLabelEmbedding

class LabelAttentionSentHead(nn.Module):
    """
    句级：句子向量 v 与每个 label 嵌入 z_l 做注意力 → 分类
    """
    def __init__(self, input_dim, num_labels,
                 label_emb: GlobalLabelEmbedding, task_name: str):
        super().__init__()
        self.label_emb = label_emb
        self.task_name = task_name
        self.num_labels = num_labels
        self.temperature = input_dim ** 0.5  # 温度参数

        # 添加投影层，将句子向量投影到标签嵌入空间
        self.sent_projection = nn.Linear(input_dim, label_emb.emb_dim)

    def forward(self, sent_vec):  # 输入 (b, d)，表示句子特征向量
        # 将句子向量投影到标签嵌入空间
        sent_vec_proj = self.sent_projection(sent_vec)  # (b, label_emb_dim)
        
        z = self.label_emb(  # 获取标签嵌入
            self.task_name,
            torch.arange(self.num_labels, device=sent_vec.device)
        )  # (C, label_emb_dim)，每个标签的嵌入

        # 点积注意力
        attn = sent_vec_proj @ z.transpose(0,1) / self.temperature  # 计算句子向量和标签向量之间的注意力得分
        return attn  # logits，表示每个标签的预测得分
