# utils/logging.py
import logging
import os
from datetime import datetime

def setup_logger(log_level=logging.INFO):
    """
    创建一个 logger, 同时将日志输出到命令行和日志文件。
    日志文件根据当前日期-时间自动命名，保存在 'log' 文件夹下。
    """
    # 日志文件名: 例如 "20231013-153045.log"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)  # 如果没有 log 文件夹，则创建
    log_filename = f"{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # 获取（或创建）一个全局的 logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 创建格式：包含时间、日志级别、信息
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    # 创建命令行 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # 创建文件 Handler
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # 如果 logger 没有 handler，则添加（避免重复添加多个 handler）
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # 输出一下日志文件路径，便于查看
    logger.info(f"Log file is saved to: {log_filepath}")

    return logger
