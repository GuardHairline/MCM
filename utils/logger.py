# utils/logger.py
import logging
import os
from datetime import datetime

def setup_logger(log_level=logging.INFO, args=None):
    """
    创建一个 logger, 同时将日志输出到命令行和日志文件。
    日志文件根据当前日期-时间自动命名，保存在 'log' 文件夹下。
    """
    # 日志文件名: 例如 "20231013-153045-masc-t-replay.log"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log"

    task_name = args.task_name
    if args.mode == "text_only":
        mode = "t"
    else:
        mode = "m"
    strategy = "none"
    if args.replay:
        strategy = "replay"
        if args.ewc:
            strategy += "-ewc"
    if args.ewc and not args.replay:
        strategy = "ewc"
    if args.si:
        strategy = "si"
    if args.lwf:
        strategy = "lwf"
    if args.mas:
        strategy = "mas"
    if args.gem:
        strategy = "gem"
    if args.pnn:
        strategy = "pnn"
    if args.tam_cl:
        strategy = "tam_cl"
    if args.moe_adapters:
        strategy  = "moe_adapters"
        if args.ddas:
            strategy += "-ddas"
    if args.use_label_embedding:
        strategy += "_label_emb"

    # 新增：根据train_text_file添加数据集标识
    dataset_id = ""
    train_file = getattr(args, 'train_text_file', '')
    if isinstance(train_file, str):
        if '2015' in train_file:
            dataset_id = "15"
        elif '2017' in train_file:
            dataset_id = "17"
        elif 'mix' in train_file.lower():
            dataset_id = "mix"

    logname = f"{timestamp}-{dataset_id}-{task_name}-{mode}-{strategy}"

    os.makedirs(log_dir, exist_ok=True)  # 如果没有 log 文件夹，则创建
    log_filename = f"{logname}.log"
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

