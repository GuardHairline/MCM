# continual/moe_adapters/moe_model_wrapper.py
import torch.nn as nn
from .moe_adapter_injector import inject_moe_adapters

class MoeAdapterWrapper(nn.Module):
    """
    包装你的 BaseMultimodalModel，使其支持任务增量的 MoE‑Adapters。
    """
    def __init__(self, base_model,
                 num_experts: int = 1,
                 top_k: int = 1):
        super().__init__()
        self.base_model = base_model
        # 冻结主干
        for p in self.base_model.parameters():
            p.requires_grad = False

        # 在文本与图像 Encoder 注入 MoE 层
        self.text_adapters  = inject_moe_adapters(base_model.text_encoder, "text",
                                                  num_experts, top_k)
        self.image_adapters = inject_moe_adapters(base_model.image_encoder, "image",
                                                  num_experts, top_k)

    # === Lifecycle ===
    def start_new_task(self):
        """在训练新任务前调用，为每层添加一个新专家。"""
        for mlist in [self.text_adapters, self.image_adapters]:
            for moe_layer in mlist:
                moe_layer.add_expert()   # freeze old & add new

    # === Forward ===
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
