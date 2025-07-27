# modules/train_refactored.py
import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# 设置文件系统共享策略，解决"Too many open files"问题
mp.set_sharing_strategy('file_system')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*byte fallback option which is not implemented in the fast tokenizers.*",
    category=UserWarning,
    module="transformers.convert_slow_tokenizer"
)
warnings.filterwarnings(
    "ignore",
    message=".*TypedStorage is deprecated.*",
    category=UserWarning,
    module="torch._utils"
)

from datasets.get_dataset import get_dataset
from modules.evaluate import evaluate_single_task, evaluate_all_learned_tasks
from .train_utils import (
    load_train_info, create_model, create_continual_learning_components,
    create_session_info, save_train_info, create_optimizer, create_ddas_optimizer
)
from .training_loop import train_model, update_continual_learning_components
from .parser import parse_train_args
from continual.label_embedding import (
    build_global_label_mapping, create_label_groups, get_label_text_mapping, generate_label_embeddings, GlobalLabelEmbedding
)
from continual.label_embedding_manager import LabelEmbeddingManager
from continual.metrics import ContinualMetrics
from utils.logging import setup_logger
from utils.ensureFileExists import ensure_directory_exists
import json
import argparse


def train(args, logger, all_tasks=[]):
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
    
    # ========== 1.5) 加载任务配置文件（如果提供）用于0样本检测 ==========
    task_config = None
    future_tasks = []
    if hasattr(args, 'task_config_file') and args.task_config_file:
        logger.info(f"Loading task configuration from: {args.task_config_file}")
        with open(args.task_config_file, 'r', encoding='utf-8') as f:
            task_config = json.load(f)
        
        # 找到当前任务在序列中的位置
        current_task_idx = None
        for i, task in enumerate(task_config['tasks']):
            if task['task_name'] == args.task_name and task['session_name'] == args.session_name:
                current_task_idx = i
                break
        
        if current_task_idx is not None:
            # 获取后续任务信息用于0样本检测
            future_tasks = task_config['tasks'][current_task_idx + 1:]
            logger.info(f"Found {len(future_tasks)} future tasks for zero-shot evaluation")
            for i, task in enumerate(future_tasks):
                logger.info(f"  Future task {i+1}: {task['task_name']} ({task['session_name']})")
    
    # ========== 2) 初始化标签嵌入管理器 ==========
    label_embedding_manager = None
    if args.use_label_embedding:
        logger.info("Initializing label embedding manager")
        # 自动生成label embedding（如不存在）
        if not args.label_embedding_path or not os.path.exists(args.label_embedding_path):
            logger.info("No existing label embedding found, generating with deberta-v3-base")
            label2idx = build_global_label_mapping()
            label_texts = get_label_text_mapping()
            pretrained_embeddings = generate_label_embeddings(
                label_texts, emb_dim=args.label_emb_dim, device="cuda" if torch.cuda.is_available() else "cpu"
            )
            label_groups = create_label_groups()
            gle = GlobalLabelEmbedding(
                label2idx=label2idx,
                emb_dim=args.label_emb_dim,
                label_groups=label_groups,
                use_similarity_regularization=args.use_similarity_reg,
                similarity_weight=args.similarity_weight,
                pretrained_embeddings=pretrained_embeddings
            )
            gle.export(args.label_embedding_path)
            logger.info(f"Label embedding generated and saved to {args.label_embedding_path}")
        # 正常加载
        label_embedding_manager = LabelEmbeddingManager(
            emb_dim=args.label_emb_dim,
            use_similarity_regularization=args.use_similarity_reg,
            similarity_weight=args.similarity_weight
        )
        label_embedding_manager.create_or_load_embedding(args.label_embedding_path, device)
        label_embedding_manager.print_label_mapping()

        # 冻结旧任务标签
        emb_obj = label_embedding_manager.get_embedding()
        if emb_obj is not None:
            emb_obj.freeze_seen_labels(args.task_name, args.num_labels)
    
    # ========== 3) 创建模型 ==========
    logger.info("Creating model")
    full_model = create_model(args, device, label_embedding_manager, logger)
    
    # ========== 3.5) 为所有任务创建模型头 ==========
    if all_tasks is not None:
        for task in all_tasks:
            session_name = task['session_name']
            task_name = task['task_name']
            if session_name in full_model.task_heads:
                continue
            task_args = argparse.Namespace(**task)
            # 选择 head 构造函数
            if getattr(task_args, 'use_label_embedding', False):
                from models.task_heads.get_head_new import get_head
            else:
                from models.task_heads.get_head import get_head
            label_emb = label_embedding_manager.get_embedding() if label_embedding_manager else None
            # 选择 base_model
            if hasattr(full_model, 'base_model'):
                base_model_for_head = full_model.base_model
                if hasattr(base_model_for_head, 'base_model'):
                    base_model_for_head = base_model_for_head.base_model
            else:
                base_model_for_head = None
            head = get_head(task_name, base_model_for_head, task_args, label_emb=label_emb)
            full_model.add_task_head(session_name, task_name, head, task_args)
        logger.info(f"All task heads created: {list(full_model.task_heads.keys())}")
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
    
    # 在服务器环境中减少worker数量以避免文件描述符问题
    num_workers = min(args.num_workers, 2) if os.environ.get('SERVER_ENV') else args.num_workers
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )
    
    # ========== 6) 创建优化器和调度器 ==========
    optimizer, scheduler = create_optimizer(full_model, args)
    
    # ========== 7) 训练模型 ==========
    logger.info("Starting training")
    train_result = train_model(
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
    # train_result 是 dict，包含所有需要的内容
    
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
    current_metrics = train_result["final_test_metrics"]
    logger.info(f"Current task metrics: {current_metrics['acc']}")
    
    # ========== 10.5) 0样本检测后续任务 ==========
    zero_shot_metrics = {}
    if future_tasks:
        logger.info("Performing zero-shot evaluation on future tasks...")
        for future_task in future_tasks:
            session_name = future_task['session_name']
            task_name = future_task['task_name']
            logger.info(f"Evaluating zero-shot performance on session: {session_name} (task: {task_name})")
            
            try:
                # 创建未来任务的参数对象
                future_args = argparse.Namespace(**future_task)
                future_args.task_name = task_name
                future_args.session_name = session_name
                
                # 对于0样本检测，不使用特定的session_name，而是使用当前模型的默认头
                # 或者创建一个临时的任务头
                if hasattr(full_model, 'set_active_head'):
                    # 尝试使用当前任务的session_name作为默认头
                    try:
                        full_model.set_active_head(session_name)
                    except:
                        # 如果失败，不设置活动头，使用默认行为
                        logger.info(f"Using default head for zero-shot evaluation on {session_name}")
                
                # 为CLAP4CLIP模型设置当前任务
                if hasattr(full_model, 'set_current_task'):
                    try:
                        full_model.set_current_task(args.session_name)
                    except:
                        logger.info(f"Using default task for zero-shot evaluation on {session_name}")
                
                # 0样本评估
                try:
                    zero_shot_acc = evaluate_single_task(full_model, task_name, "test", device, future_args)
                    zero_shot_metrics[session_name] = zero_shot_acc
                    logger.info(f"Zero-shot accuracy on {session_name}: {zero_shot_acc['acc']:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to evaluate zero-shot performance on {session_name}: {e}")
                    # 如果评估失败，记录一个默认值
                    zero_shot_metrics[session_name] = {"acc": 0.0, "micro_prec": 0.0, "micro_recall": 0.0, "micro_f1": 0.0}
                    logger.info(f"Zero-shot accuracy on {session_name}: 0.0000 (fallback)")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate zero-shot performance on {session_name}: {str(e)}")
                zero_shot_metrics[session_name] = None
        
        # # 将0样本指标添加到session_info
        # session_info["zero_shot_metrics"] = zero_shot_metrics
    
    # ========== 11) 更新训练信息 ==========
    session_info["details"].update({
        "epoch_losses": train_result["epoch_losses"],
        "dev_metrics_history": train_result["dev_metrics_history"],
        "final_dev_metrics": train_result["final_dev_metrics"],
        "final_test_metrics": train_result["final_test_metrics"]
    })
    
    # 获取当前任务的索引（基于已学习的任务数量）
    task_idx = len(train_info["tasks"])
    
    # 更新准确率矩阵
    cm = ContinualMetrics()
    cm.acc_matrix = train_info["acc_matrix"]
    
    # 构建性能列表：包含所有已学习任务的准确率
    performance_list = []
    
    # 如果有之前学习的任务，需要评估所有任务
    if old_sessions_count > 0:
        logger.info(f"Previous sessions: {[s.get('session_name', 'unknown') for s in train_info['sessions']]}")
        all_metrics = evaluate_all_learned_tasks(full_model, train_info["sessions"], device, train_info)
        logger.info(f"All tasks metrics: {all_metrics}")
        logger.info(f"Current task metrics: {current_metrics['acc']}")
        # 将当前任务的准确率添加到列表中
        performance_list = all_metrics + [current_metrics["acc"]]
        logger.info(f"Final performance list: {performance_list}")
    else:
        # 第一个任务，只有当前任务的准确率
        performance_list = [current_metrics["acc"]]
        logger.info(f"First task performance list: {performance_list}")
    
    # 将0样本指标传递给准确率矩阵
    cm.update_acc_matrix(task_idx, performance_list, zero_shot_metrics)
    train_info["acc_matrix"] = cm.acc_matrix
    
    # 添加当前任务到训练信息中
    train_info["sessions"].append(session_info)
    train_info["tasks"].append(args.task_name)
    
    # ========== 12) 计算持续学习指标 ==========
    # 若是第一个任务, 不算持续学习指标
    if len(train_info["sessions"]) <= 1:
        logger.info("[Info] This is the first task, skip any CL metrics.")
        final_metrics = {}
    else:
        k = len(train_info["sessions"])  # 总任务数
        from continual.metrics import compute_multimodal_transfer_metrics, analyze_task_similarity_transfer
        
        # 获取任务名称列表
        task_names = [session.get('task_name', 'unknown') for session in train_info["sessions"]]
        
        # 计算多模态转移指标
        final_metrics = compute_multimodal_transfer_metrics(cm, k, task_names)
        
        # 分析任务相似性转移
        similarity_analysis = analyze_task_similarity_transfer(cm, task_names)
        if similarity_analysis:
            final_metrics.update(similarity_analysis)
            logger.info(f"Task similarity analysis: {similarity_analysis}")
        
        logger.info(f"Continual Metrics after learning {k} tasks: {final_metrics}")
    
    # 合并训练指标和持续学习指标
    session_info["final_metrics"] = {
        "best_metrics": train_result["best_metrics"],
        "continual_metrics": final_metrics
    }
    
    # ========== 13) 保存模型 ==========
    logger.info(f"Saving model to: {args.output_model_path}")
    torch.save(full_model.state_dict(), args.output_model_path)
    
    # 保存任务头信息
    task_heads_path = args.output_model_path.replace('.pt', '_task_heads.pt')
    if hasattr(full_model, 'save_task_heads'):
        full_model.save_task_heads(task_heads_path)
        logger.info(f"Task heads saved to: {task_heads_path}")
    
    logger.info(f"Model saved successfully to: {args.output_model_path}")
    
    # 保存训练信息
    save_train_info(train_info, args.train_info_json, logger)
    
    logger.info(f"=== Training completed for task: {args.task_name} ===")
    return train_result["best_metrics"]


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