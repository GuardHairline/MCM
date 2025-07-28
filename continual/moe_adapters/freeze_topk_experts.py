# continual/moe_adapters/freeze_topk_experts.py
from collections import Counter

def freeze_topk_experts(model, k: int):
    """
    冻结模型中所有 MoE 层激活频次最高的 k 个专家。
    model: Full_Model
    k: 每层冻结的专家数量
    """
    # 访问封装在 MoeAdapterWrapper 中的 adapters
    # 适配 text 和 image 模态
    moe_modules = []
    if hasattr(model.base_model, 'text_adapters'):
        moe_modules += model.base_model.text_adapters
    if hasattr(model.base_model, 'image_adapters'):
        moe_modules += model.base_model.image_adapters

    for layer in moe_modules:
        if hasattr(layer, 'activation_counter'):
            # 按激活次数排序
            sorted_experts = sorted(
                layer.activation_counter.items(), key=lambda item: item[1], reverse=True
            )
            # 取前 k 个
            freeze_ids = [idx for idx, _ in sorted_experts[:k]]
            # 冻结权重
            for idx, expert in enumerate(layer.experts):
                if idx in freeze_ids:
                    for p in expert.parameters():
                        p.requires_grad = False
            # 清空计数器，供下一个任务重新统计
            layer.activation_counter.clear()
        else:
            # 未找到统计信息，不进行冻结
            continue
