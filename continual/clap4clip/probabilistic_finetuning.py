import torch
import torch.nn as nn

class ProbabilisticFinetuning(nn.Module):
    def __init__(self, model, num_tasks, finetune_lambda=0.1):
        super(ProbabilisticFinetuning, self).__init__()
        self.model = model
        self.num_tasks = num_tasks
        self.finetune_lambda = finetune_lambda

    def forward(self, x):
        # 概率微调
        task_weights = torch.softmax(torch.randn(self.num_tasks), dim=-1)  # 任务权重
        logits = self.model(x)
        return logits, task_weights
