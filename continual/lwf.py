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
        self.old_model = old_model.eval()  # teacher 固定
        self.T = T
        self.alpha = alpha

    def distillation_loss(self, new_logits, inputs):
        """
        new_logits: tensor, shape either (B, C) for sentence or (B, L, C) for token tasks
        inputs: dict with keys input_ids, attention_mask, token_type_ids, image_tensor
        """
        with torch.no_grad():
            # --- 1) 先算 teacher logits，shape 与 new_logits 保持一致 ---
            if new_logits.dim() == 3:
                # logger.warning("Token-level distillation.")
                # token-level: 强制走 sequence=True 路径
                feat = self.old_model.base_model(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs.get("token_type_ids", None),
                    inputs["image_tensor"],
                    return_sequence=True
                )
                old_logits = self.old_model.head(feat)  # (B, L, C)
            else:
                # logger.info("Sentence-level distillation.")
                # sentence-level
                old_logits = self.old_model(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs.get("token_type_ids", None),
                    inputs["image_tensor"]
                )  # (B, C)

        # --- 2) flatten 到 (N, C) 计算 KL ---
        C = new_logits.size(-1)
        new_flat = new_logits.view(-1, C)
        old_flat = old_logits.view(-1, C)

        new_log = F.log_softmax(new_flat / self.T, dim=-1)
        old_soft = F.softmax(    old_flat / self.T, dim=-1)

        kd = F.kl_div(new_log, old_soft, reduction='batchmean') * (self.T ** 2)
        return self.alpha * kd
