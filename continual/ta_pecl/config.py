# continual/ta_pecl/config.py

# 映射表保持不变
TASK_NAME_MAP = {"masc": 0, "mate": 1, "mner": 2, "mabsa": 3}
MODE_NAME_MAP = {"text_only": 0, "multimodal": 1}

def get_expert_config(hidden_size=768, r=8):
    configs = {}
    
    # --- 1. 任务专家 (标记其所属的任务ID) ---
    for t, tid in TASK_NAME_MAP.items():
        configs[f"expert_{t}"] = {
            "dim": hidden_size, "r": r, 
            "init_task_id": tid  # <--- 关键元数据：初始化时偏向哪个任务
        }

    # --- 2. 模态专家 (标记其所属的模态ID) ---
    configs["expert_text"] = {"dim": hidden_size, "r": r, "init_mode_id": 0}
    configs["expert_multi"] = {"dim": hidden_size, "r": r, "init_mode_id": 1}

    # --- 3. DEQA专家 (标记其所属的模态ID) ---
    configs["expert_deqa"] = {"dim": hidden_size, "r": r, "init_mode_id": 1}

    # --- 4. 灵活专家 (无预设) ---
    num_flexible = 4
    for i in range(num_flexible):
        configs[f"expert_flex_{i}"] = {"dim": hidden_size, "r": r}

    return configs

# 建议 Top-K 设为 3 或 4
# 因为现在所有专家都在同一个池子里竞争，Top-K 控制了稀疏度