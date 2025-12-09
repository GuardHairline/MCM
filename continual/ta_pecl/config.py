# continual/ta_pecl/config.py

# 任务 ID 映射表 (必须固定)
TASK_NAME_MAP = {
    "masc": 0,
    "mate": 1,
    "mner": 2,
    "mabsa": 3
}

def get_expert_config(hidden_size=768, r=8):
    configs = {}
    
    # --- 1. 任务特定专家 (Task-Specific Experts) ---
    # 初始化意图：希望能分别处理这四类任务的独特逻辑
    for t in TASK_NAME_MAP.keys():
        configs[f"expert_{t}"] = {"type": "task", "desc": f"Expert init for {t}"}

    # --- 2. 模态特定专家 (Modality Experts) ---
    configs["expert_text"] = {"type": "modality", "desc": "Text-only logic"}
    configs["expert_multi"] = {"type": "modality", "desc": "Cross-modal interaction"}

    # --- 3. DEQA 描述增强专家 (Description Expert) ---
    # 这个专家旨在专门消化由 Image Description 带来的额外信息
    configs["expert_deqa"] = {"type": "aux", "desc": "Processes image description features"}

    # --- 4. 灵活专家 (Flexible Experts) ---
    # 捕捉通用语法或底层特征，弥补上述专家的不足
    num_flexible = 3
    for i in range(num_flexible):
        configs[f"expert_flex_{i}"] = {"type": "flexible", "desc": "General purpose"}

    return configs

# 总共 4+2+1+3 = 10 个专家
# 建议 Top-K 设为 3 或 4