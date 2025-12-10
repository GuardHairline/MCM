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

    task_name = args.task_name if args else "unknown"
    
    # 模式简写
    if args and hasattr(args, 'mode'):
        mode = "t" if args.mode == "text_only" else "m"
    else:
        mode = "u" # unknown

    # =========================================================
    # 动态生成策略名称
    # 维护一个 (参数名, 日志缩写) 的列表，按优先级排序
    # =========================================================
    method_map = [
        ('ta_pecl', 'ta_pecl'),       # 新增的方法放前面或后面都可以
        ('deqa', 'deqa'),
        ('moe_adapters', 'moe'),
        ('replay', 'replay'),
        ('ewc', 'ewc'),
        ('gem', 'gem'),
        ('lwf', 'lwf'),
        ('si', 'si'),
        ('mas', 'mas'),
        ('pnn', 'pnn'),
        ('tam_cl', 'tam'),
        ('ddas', 'ddas'),
        ('clap4clip', 'clap'),
        ('mymethod', 'my'),
        ('use_label_embedding', 'lbl'),
        # 在这里添加新方法...
    ]

    active_strategies = []
    if args:
        for arg_name, short_name in method_map:
            # 检查参数是否存在且为真 (True, 1, 非空对象等)
            val = getattr(args, arg_name, None)
            if val:
                # 特殊处理: 如果是 int 类型的开关，值为 0 则不添加
                if isinstance(val, int) and val == 0:
                    continue
                active_strategies.append(short_name)

    # 如果没有任何策略激活，标记为 none
    if not active_strategies:
        strategy = "none"
    else:
        # 用下划线连接所有激活的策略，例如 "replay_ewc_lbl"
        strategy = "_".join(active_strategies)

    # =========================================================

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

    # 组装最终文件名
    # 格式: 时间-数据集-任务-模式-策略组合.log
    parts = [timestamp]
    if dataset_id: parts.append(dataset_id)
    parts.append(task_name)
    parts.append(mode)
    parts.append(strategy)
    
    logname = "-".join(parts)

    os.makedirs(log_dir, exist_ok=True)  # 如果没有 log 文件夹，则创建
    log_filename = f"{logname}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # 获取（或创建）一个全局的 logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 清除旧的 handlers 避免重复打印
    if logger.hasHandlers():
        logger.handlers.clear()
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

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 输出一下日志文件路径，便于查看
    logger.info(f"Log file is saved to: {log_filepath}")

    return logger

