# continual/moe_adapters/moe_adapter_injector.py
"""
在 HuggingFace Transformer block 中插入 MoEAdapterLayer。
"""
import torch.nn as nn
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Layer
from transformers.models.vit.modeling_vit import ViTLayer
from .moe_adapter_layer import MoEAdapterLayer

def _inject_into_block(block: nn.Module, hidden_size: int,
                       num_experts: int, top_k: int):
    moe_layer = MoEAdapterLayer(hidden_size, num_experts, top_k)
    # 在 FFN 之后插入
    old_forward = block.forward
    def new_forward(*args, **kwargs):
        out = old_forward(*args, **kwargs)
        if isinstance(out, tuple):
            # ViT 返回 (x, att)
            x, *rest = out
            x = moe_layer(x)
            return (x, *rest)
        else:
            return moe_layer(out)
    block.forward = new_forward
    block.moe_adapter = moe_layer        # 便于之后 add_expert
    return moe_layer

def inject_moe_adapters(model: nn.Module,
                        encoder_type: str,
                        num_experts: int = 1,
                        top_k: int = 1):
    """
    遍历 DeBERTa/Vit encoder 所有 Transformer block，插入 MoEAdapterLayer。
    返回 adapter 列表，用于统一管理。
    """
    adapters = []
    if encoder_type == "text":
        for blk in model.encoder.layer:        # DeBERTa v3
            adapters.append(_inject_into_block(blk, model.config.hidden_size,
                                               num_experts, top_k))
    elif encoder_type == "image":
        for blk in model.encoder.layer:        # ViT
            adapters.append(_inject_into_block(blk, model.config.hidden_size,
                                               num_experts, top_k))
    else:
        raise ValueError
    return adapters
