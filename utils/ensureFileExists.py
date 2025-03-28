import os
import logging
logger = logging.getLogger("info")

def ensure_directory_exists(directory):
    """确保指定的目录存在，不存在则创建目录"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")