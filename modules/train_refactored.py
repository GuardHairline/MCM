# modules/train_refactored.py
import os
import torch
from torch.utils.data import DataLoader

from datasets.get_dataset import get_dataset
from modules.evaluate import evaluate_single_task, evaluate_all_learned_tasks
from .train_utils import (
    load_train_info, create_model, create_continual_learning_components,
    create_session_info, save_train_info, create_optimizer, create_ddas_optimizer
)
from .training_loop import train_model, update_continual_learning_components
from .parser import parse_train_args
from continual.label_embedding_manager import LabelEmbeddingManager
from continual.metrics import ContinualMetrics
from utils.logging import setup_logger
from utils.ensureFileExists import ensure_directory_exists


def train(args, logger):
    """主训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确保目录存在
    ensure_directory_exists(args.train_info_json)
    ensure_directory_exists(args.ewc_dir)
    ensure_directory_exists(args.output_model_path)
    
    logger.info(f"=== Start training for new task: {args.task_name} ===")
    
    # ========== 1) 加载训练信息 ==========
    train_info = load_train_info(args.train_info_json)
    old_sessions_count = len(train_info["sessions"])
    logger.info(f"Previously learned sessions: {old_sessions_count}")
    
    # ========== 2) 初始化标签嵌入管理器 ==========
    label_embedding_manager = None
    if args.use_label_embedding:
        logger.info("Initializing label embedding manager")
        label_embedding_manager = LabelEmbeddingManager(
            emb_dim=getattr(args, 'label_emb_dim', 128),
            use_similarity_regularization=getattr(args, 'use_similarity_reg', True),
            similarity_weight=getattr(args, 'similarity_weight', 0.1)
        )
        
        # 创建或加载标签嵌入
        embedding_path = getattr(args, 'label_embedding_path', None)
        label_embedding_manager.create_or_load_embedding(embedding_path, device)
        
        # 打印标签映射信息
        label_embedding_manager.print_label_mapping()
    
    # ========== 3) 创建模型 ==========
    logger.info("Creating model")
    full_model = create_model(args, device, label_embedding_manager)
    
    # ========== 4) 创建持续学习组件 ==========
    logger.info("Creating continual learning components")
    ewc, fisher_selector, replay_memory, lwf, si, mas, gem, pnn = create_continual_learning_components(
        args, full_model, train_info, device, logger
    )
    
    # ========== 5) 加载数据 ==========
    logger.info("Loading datasets")
    train_dataset = get_dataset(args.task_name, "train", args)
    val_dataset = get_dataset(args.task_name, "dev", args)
    test_dataset = get_dataset(args.task_name, "test", args)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    # ========== 6) 创建优化器和调度器 ==========
    optimizer, scheduler = create_optimizer(full_model, args)
    
    # ========== 7) 训练模型 ==========
    logger.info("Starting training")
    best_metrics = train_model(
        model=full_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
        ewc=ewc,
        fisher_selector=fisher_selector,
        replay_memory=replay_memory,
        lwf=lwf,
        si=si,
        mas=mas,
        gem=gem,
        label_embedding_manager=label_embedding_manager,
        logger=logger
    )
    
    # ========== 8) 保存标签嵌入 ==========
    if label_embedding_manager and args.label_embedding_path:
        label_embedding_manager.save_embedding(args.label_embedding_path)
    
    # ========== 9) 更新持续学习组件 ==========
    session_info = create_session_info(args)
    session_info = update_continual_learning_components(
        full_model, train_loader, device, args,
        ewc, fisher_selector, si, gem, session_info, logger
    )
    
    # ========== 10) 评估和更新训练信息 ==========
    logger.info("Evaluating model")
    
    # 评估当前任务
    current_metrics = evaluate_single_task(full_model, args.task_name, "test", device, args)
    logger.info(f"Current task metrics: {current_metrics}")
    
    # 评估所有已学习的任务
    if old_sessions_count > 0:
        all_metrics = evaluate_all_learned_tasks(full_model, test_loader, device, args)
        logger.info(f"All tasks metrics: {all_metrics}")
    
    # ========== 11) 更新训练信息 ==========
    session_info["final_metrics"] = best_metrics
    
    train_info["sessions"].append(session_info)
    train_info["tasks"].append(args.task_name)
    
    # 更新准确率矩阵
    cm = ContinualMetrics()
    cm.acc_matrix = train_info["acc_matrix"]
    cm.update_acc_matrix(args.task_name, current_metrics)
    train_info["acc_matrix"] = cm.acc_matrix
    
    # 保存训练信息
    save_train_info(train_info, args.train_info_json)
    
    logger.info(f"=== Training completed for task: {args.task_name} ===")
    return best_metrics


def main():
    """主函数"""
    # 使用新的parser模块
    args = parse_train_args()
    
    # 设置日志
    logger = setup_logger(args=args)
    
    # 开始训练
    try:
        best_metrics = train(args, logger)
        logger.info(f"Training completed successfully. Best metrics: {best_metrics}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 