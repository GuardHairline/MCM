# modules/common_keys.py – 新建
def build_task_key(task_name: str, mode: str):
    """例如 ('mabsa', 'multi') → 'mabsa_multi'"""
    return f"{task_name.lower()}_{mode.lower()}"
