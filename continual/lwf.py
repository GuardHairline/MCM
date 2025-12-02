# continual/lwf.py

import torch
import torch.nn.functional as F
import logging
logger = logging.getLogger("lwf")

class LwFDistiller:
    """
    Learning without Forgetting for both sentence- and token-level tasks.
    """
    def __init__(self, old_model, T=2.0, alpha=0.5):
        """
        Args:
            old_model: 上一轮训练好的模型（已冻结）
            T: 温度系数
            alpha: 蒸馏损失权重
        """
        self.old_model = old_model.eval()  # teacher 固定
        for p in self.old_model.parameters():
            p.requires_grad = False
        self.T = T
        self.alpha = alpha

    def distillation_loss(self, new_logits, inputs):
        """
        计算 LwF 损失：遍历所有旧任务，让新旧模型都使用旧任务的 Head 进行预测，并计算 KL 散度。
        
        Args:
            current_model: 当前正在训练的模型
            inputs: 当前 batch 数据
            previous_task_names: 历史任务名称列表 (e.g. ["mner_1", "mate_2"])
        """
        total_distill_loss = 0.0
        
        if not previous_task_names:
            return torch.tensor(0.0).to(inputs['input_ids'].device)

        # 获取 Encoder 特征 (避免重复计算)
        # 注意：如果使用了 DDAS 或 MoE，可能需要小心特征是否一致，这里假设 Backbone 输出特征可复用
        # 为了严谨，我们让模型自己处理 forward
        
        for task_name in previous_task_names:
            # 1. 让旧模型使用旧 Head 预测
            self.old_model.set_active_head(task_name)
            with torch.no_grad():
                # 调用 model 的 forward，它会自动使用 set_active_head 设置的 head
                # 注意：inputs 解包
                if hasattr(self.old_model, 'base_model'): # 兼容您的架构
                     # 这里假设 forward 会调用 head
                     # 如果您的 model forward 逻辑依赖当前 active head
                     old_logits = self.old_model(
                         inputs['input_ids'], 
                         inputs['attention_mask'], 
                         inputs.get('token_type_ids'), 
                         inputs['image_tensor']
                     )
                     # 如果 model forward 返回的是 tuple (loss, logits)，取 logits
                     if isinstance(old_logits, tuple):
                         old_logits = old_logits[1]

            # 2. 让当前新模型使用旧 Head 预测 (这是为了让新模型"记住"旧任务的映射关系)
            # 这是一个关键点：新模型不仅要学新任务，还要在旧 Head 上模仿旧模型的行为
            current_model.set_active_head(task_name)
            new_logits_on_old_task = current_model(
                 inputs['input_ids'], 
                 inputs['attention_mask'], 
                 inputs.get('token_type_ids'), 
                 inputs['image_tensor']
            )
            if isinstance(new_logits_on_old_task, tuple):
                new_logits_on_old_task = new_logits_on_old_task[1]

            # 3. 计算 KL 散度
            # 展平 (Batch * Seq, Num_Labels) 以适配 token-level
            num_labels = old_logits.size(-1)
            old_flat = old_logits.view(-1, num_labels)
            new_flat = new_logits_on_old_task.view(-1, num_labels)

            # Softmax 温度缩放
            old_soft = F.softmax(old_flat / self.T, dim=-1)
            new_log_soft = F.log_softmax(new_flat / self.T, dim=-1)

            # KLDivLoss
            loss = F.kl_div(new_log_soft, old_soft, reduction='batchmean') * (self.T ** 2)
            total_distill_loss += loss

        # 恢复当前模型到当前任务 Head，以免影响后续的 CE Loss 计算
        # (假设外部循环知道当前任务名，或者由外部负责切回)
        
        return self.alpha * total_distill_loss
