import os
import logging
logger = logging.getLogger("info")


def ensure_directory_exists(path):
    """确保指定路径的父目录存在。若路径为目录，则直接创建该目录。"""
    # 规范化路径，处理可能的路径分隔符问题
    normalized_path = os.path.normpath(path)

    # 判断路径是否存在
    if os.path.exists(normalized_path):
        if os.path.isfile(normalized_path):
            # 存在且为文件，获取父目录
            parent_dir = os.path.dirname(normalized_path)
        else:
            # 存在且为目录，无需操作
            return
    else:
        # 路径不存在，判断是否为目录路径
        # 若路径以分隔符结尾或无法推断为文件，则视为目录
        is_dir = normalized_path.endswith(os.sep) or not os.path.splitext(normalized_path)[1]
        if is_dir:
            # 视为目录路径，直接创建该目录
            parent_dir = normalized_path
        else:
            # 视为文件路径，创建父目录
            parent_dir = os.path.dirname(normalized_path)

    # 创建目录（若不存在）
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        logger.info(f"Created directory: {parent_dir}")