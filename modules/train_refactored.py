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
    create_session_info, save_train_info, create_optimizer, create_ddas_optimizer,
    create_scheduler
)
from .training_loop_fixed import train_model, update_continual_learning_components
from .parser import parse_train_args
from continual.label_embedding import (
    build_global_label_mapping, create_label_groups, get_label_text_mapping, generate_label_embeddings, GlobalLabelEmbedding
)
from continual.label_embedding_manager import LabelEmbeddingManager
from continual.moe_adapters.freeze_topk_experts import freeze_topk_experts
from continual.metrics import ContinualMetrics
from utils.logger import setup_logger
from utils.ensureFileExists import ensure_directory_exists
from visualize.feature_clustering import visualize_task_after_training, visualize_all_tasks_evolution
from visualize.feature_clustering_enhanced import visualize_task_enhanced
from visualize.training_curves import plot_training_curves
import json
import argparse


def train(args, logger, all_tasks=[]):
    """主训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 提取配置文件名（用于可视化文件命名，避免不同配置互相覆盖）
    config_name = None
    if hasattr(args, 'task_config_file') and args.task_config_file:
        from pathlib import Path
        config_name = Path(args.task_config_file).stem  # 提取文件名（不含路径和扩展名）
        logger.info(f"配置文件名: {config_name}")
    
    # 确保目录存在
    ensure_directory_exists(args.train_info_json)
    ensure_directory_exists(args.ewc_dir)
    ensure_directory_exists(args.output_model_path)
    
    logger.info(f"=== Start training for new task: {args.task_name} ===")
    
    # ========== 1) 加载训练信息 ==========
    train_info = load_train_info(args.train_info_json)
    old_sessions_count = len(train_info["sessions"])
    args.old_sessions_count = old_sessions_count
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

    # 注册当前任务的头（确保优化器/切换可用）
    current_head_key = getattr(args, 'head_key', args.session_name)
    # 若共享头已存在（通过 head_key），不重复注册
    if not (full_model.head_manager.has_head(args.session_name) or full_model.head_manager.has_head(current_head_key)):
        full_model.add_task_head(args.session_name, args.task_name, full_model.head, args)
    # 设置当前活动头
    try:
        full_model.set_active_head(args.session_name, strict=False)
    except Exception:
        pass
    
    # ========== 3.5) 只为历史任务创建模型头（延迟创建模式） ==========
    # 注意：不再为未来任务预创建head，只在需要时创建
    if all_tasks is not None and not args.tam_cl:
        # 只为已经学习过的任务加载head（从train_info中获取）
        learned_sessions = set(s['session_name'] for s in train_info.get('sessions', []))
        logger.info(f"Loading heads for {len(learned_sessions)} previously learned sessions")
        
        for task in all_tasks:
            session_name = task['session_name']
            task_name = task['task_name']
            
            # 只处理历史任务
            if session_name not in learned_sessions:
                continue
            
            # 如果head已存在，跳过
            if full_model.head_manager.has_head(session_name):
                logger.debug(f"Head for {session_name} already exists, skipping")
                continue
            
            try:
                task_args = argparse.Namespace(**task)
                if not hasattr(task_args, 'head_key'):
                    task_args.head_key = session_name
                use_label_embedding = getattr(task_args, 'use_label_embedding', False)
                
                # 使用TaskHeadManager创建head
                logger.info(f"Creating head for historical task: {session_name} ({task_name})")
                head_key = getattr(task_args, 'head_key', session_name)
                head = full_model.head_manager.create_and_register_head(
                    session_name, task_name, task_args, use_label_embedding, head_key=head_key
                )
                
                if head is None:
                    logger.warning(f"Failed to create head for {session_name}")
                    
            except Exception as e:
                logger.warning(f"Error creating head for {session_name}: {e}")
                continue
        
        logger.info(f"Historical task heads loaded: {full_model.head_manager.get_head_count()}")
    elif args.tam_cl:
        logger.info("TAM-CL: Using task-specific adapters instead of separate heads")
    
    # ========== 3.6) MoE-Adapters: 为新任务添加专家 ==========
    if args.moe_adapters:
        logger.info("MoE-Adapters: Adding new expert for current task")
        # 检查是否是第一个任务
        is_first_task = len(train_info.get('sessions', [])) == 0
        
        if is_first_task:
            logger.info("  First task: Expert already created during model initialization")
        else:
            logger.info(f"  Task {len(train_info['sessions']) + 1}: Calling start_new_task() to add new expert")
            # 调用MoeAdapterWrapper的start_new_task方法
            if hasattr(full_model.base_model, 'start_new_task'):
                full_model.base_model.start_new_task()
                logger.info("  ✓ New expert added and old experts frozen")
            else:
                logger.warning("  ✗ base_model does not have start_new_task method!")
    
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
    # ========== 5.5) 如果使用 GEM，注册当前任务的记忆样本 ==========
    if gem is not None:
        gem.register_task(args.task_name, train_dataset)
    # ========== 6) 创建优化器和调度器 ==========
    optimizer = create_optimizer(full_model, args)
    total_training_steps = len(train_loader) * args.epochs if len(train_loader) > 0 else args.epochs
    scheduler = create_scheduler(optimizer, args, total_training_steps)
    
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
        ewc, fisher_selector, si, mas, gem, session_info, logger
    )
    if args.moe_adapters and hasattr(full_model.base_model, 'text_adapters'):
    # 假设需要冻结每层一个专家，可用 args.freeze_topk_experts 参数配置
        freeze_topk = getattr(args, 'freeze_topk_experts', 1)
        freeze_topk_experts(full_model, freeze_topk)
    # ========== 10) 评估和更新训练信息 ==========
    logger.info("Evaluating model")
    # 评估当前任务（使用DEV集作为主要指标，TEST集仅用于记录）
    current_dev_metrics = train_result["final_dev_metrics"]
    current_test_metrics = train_result["final_test_metrics"]
    logger.info(f"Current task DEV metrics: {current_dev_metrics['acc']:.4f}")
    logger.info(f"Current task TEST metrics (reference only): {current_test_metrics['acc']:.4f}")
    
    # ========== 10.5) 0样本检测后续任务（使用DEV集，不使用TEST集） ==========
    zero_shot_metrics = {}
    if future_tasks:
        logger.info("Performing zero-shot evaluation on future tasks (using DEV set)...")
        logger.info("⚠️  IMPORTANT: Creating temporary task heads with random weights for zero-shot evaluation")
        logger.info("   (Different tasks have different label spaces, cannot use current task's head!)")
        
        # ✅ 修复完成：DEQA现在与框架完全兼容
        # DEQA使用：DEQA专家融合特征 + TaskHead输出logits
        # 普通模型使用：BaseModel特征 + TaskHead输出logits
        # 两者都使用相同的head_manager机制！
        
        from models.deqa_expert_model import DEQAMultimodalModel
        is_deqa = isinstance(full_model, DEQAMultimodalModel)
        if is_deqa:
            logger.info("   (DEQA模型: 使用DEQA专家融合特征 + 临时随机head)")
        
        for future_task in future_tasks:
            session_name = future_task['session_name']
            task_name = future_task['task_name']
            logger.info(f"Zero-shot evaluation on: {session_name} (task: {task_name})")
            
            try:
                # 创建未来任务的参数对象
                future_args = argparse.Namespace(**future_task)
                if not hasattr(future_args, 'head_key'):
                    future_args.head_key = session_name
                future_args.task_name = task_name
                future_args.session_name = session_name
                
                # 🔑 关键：为未来任务临时创建一个随机初始化的head
                # 原因：不同任务的标签空间不同！
                # 例如：MASC的0=NEG，但MATE的0=O，含义完全不同
                logger.info(f"  Step 1: Creating temporary random head for {session_name}")
                logger.info(f"          Task: {task_name}, Labels: {future_args.num_labels}")
                
                # 检查是否已经存在这个head
                head_exists = full_model.head_manager.has_head(session_name)
                
                if not head_exists:
                    # 创建临时head（随机初始化）
                    # ✓ 对于DEQA：创建DEQA专家集成 + 临时head
                    # ✓ 对于普通模型：仅创建临时head
                    use_label_embedding = getattr(future_args, 'use_label_embedding', False)
                    
                    if is_deqa:
                        # DEQA需要先添加任务（创建专家）
                        full_model.add_task(task_name, session_name, future_args.num_labels, future_args)
                    else:
                        # 普通模型只需创建head
                        head_key = getattr(future_args, 'head_key', session_name)
                        temp_head = full_model.head_manager.create_and_register_head(
                            session_name, task_name, future_args, use_label_embedding, head_key=head_key
                        )
                        if temp_head is None:
                            logger.warning(f"  ✗ Failed to create temporary head for {session_name}")
                            zero_shot_metrics[session_name] = {"acc": 0.0, "micro_prec": 0.0, "micro_recall": 0.0, "micro_f1": 0.0}
                            continue
                    
                    logger.info(f"  ✓ Temporary head created (random weights)")
                else:
                    logger.info(f"  ✓ Head already exists for {session_name}")
                
                # 设置活动head为未来任务的head
                logger.info(f"  Step 2: Setting active head to {session_name}")
                full_model.set_active_head(session_name, strict=True)
                
                # 0样本评估（使用DEV集，不是TEST集）
                # 此时：
                # - 普通模型: 训练好的base_model + 随机head ✓
                # - DEQA: 训练好的DEQA专家融合 + 随机head ✓
                logger.info(f"  Step 3: Evaluating with trained features + random head")
                try:
                    zero_shot_acc = evaluate_single_task(full_model, task_name, "dev", device, future_args)
                    zero_shot_metrics[session_name] = zero_shot_acc
                    logger.info(f"  ✓ Zero-shot DEV accuracy on {session_name}: {zero_shot_acc['acc']:.4f}")
                except Exception as e:
                    logger.warning(f"  ✗ Failed zero-shot evaluation on {session_name}: {e}")
                    zero_shot_metrics[session_name] = {"acc": 0.0, "micro_prec": 0.0, "micro_recall": 0.0, "micro_f1": 0.0}
                    logger.info(f"  Zero-shot DEV accuracy on {session_name}: 0.0000 (fallback)")
                
                # 🔑 重要：评估完后删除临时head（节省内存）
                if not head_exists:
                    logger.info(f"  Step 4: Removing temporary head to save memory")
                    full_model.head_manager.remove_head(session_name)
                    if is_deqa:
                        # DEQA还需要删除专家
                        del full_model.deqa_cl.task_ensembles[session_name]
                    logger.info(f"  ✓ Temporary components removed")
                
            except Exception as e:
                logger.warning(f"  ✗ Error in zero-shot evaluation for {session_name}: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                zero_shot_metrics[session_name] = None
        
        # # 将0样本指标添加到session_info
        # session_info["zero_shot_metrics"] = zero_shot_metrics
    
    # ========== 11) 更新训练信息 ==========
    session_info["details"].update({
        "epoch_losses": train_result["epoch_losses"],
        "dev_metrics_history": train_result["dev_metrics_history"],
        "dev_losses": train_result.get("dev_losses", []),  # 验证loss历史
        "best_metric_summary": train_result.get("best_metric_summary", {}),  # ✨ 最佳dev指标摘要（含最佳epoch）
        "final_dev_metrics": train_result["final_dev_metrics"],  # 用于模型选择和early stopping
        "final_test_metrics": train_result["final_test_metrics"],  # 仅用于最终报告
        "dev_used_for_decisions": True,  # 标记使用DEV集进行训练决策
        "test_for_reference_only": True,  # 标记TEST集仅供最终参考
        "zero_shot_metrics": zero_shot_metrics if zero_shot_metrics else {}  # 0样本检测结果（基于DEV）
    })
    
    # ✨ 在session_info的顶层也记录最佳指标，方便访问
    if "best_metric_summary" in train_result:
        session_info["best_dev_epoch"] = train_result["best_metric_summary"].get("best_epoch", 0)
        session_info["best_dev_metric"] = train_result["best_metric_summary"].get("best_dev_metric", 0.0)
        session_info["best_dev_metric_type"] = train_result["best_metric_summary"].get("metric_type", "unknown")
    
    # 获取当前任务的索引（基于已学习的任务数量）
    task_idx = len(train_info["tasks"])
    
    # ✨ 更新准确率矩阵（支持三种指标）
    cm = ContinualMetrics()
    cm.acc_matrix = train_info.get("acc_matrix", [])
    cm.chunk_f1_matrix = train_info.get("chunk_f1_matrix", [])
    cm.token_micro_f1_no_o_matrix = train_info.get("token_micro_f1_no_o_matrix", [])
    
    # ✨ 构建三种性能列表：包含所有已学习任务的准确率
    performance_list = []  # 默认指标（acc）
    chunk_f1_list = []  # 序列任务指标1
    token_micro_f1_no_o_list = []  # 序列任务指标2
    
    # 辅助函数：从metrics中提取指定指标
    def extract_metrics_for_all_tasks(full_model, sessions, device, train_info, metric_name='acc'):
        """评估所有历史任务并提取指定指标"""
        metrics_list = []
        for session in sessions:
            session_args = argparse.Namespace(**session["args"])
            task_metrics = evaluate_single_task(full_model, session["task_name"], "test", device, session_args)
            metrics_list.append(task_metrics.get(metric_name, 0.0))
        return metrics_list
    
    # 如果有之前学习的任务，需要评估所有任务（使用TEST集进行最终评估）
    if old_sessions_count > 0:
        logger.info(f"Previous sessions: {[s.get('session_name', 'unknown') for s in train_info['sessions']]}")
        
        # 评估所有历史任务，获取三种指标
        all_acc_metrics = extract_metrics_for_all_tasks(full_model, train_info["sessions"], device, train_info, 'acc')
        all_chunk_f1_metrics = extract_metrics_for_all_tasks(full_model, train_info["sessions"], device, train_info, 'chunk_f1')
        all_token_micro_f1_no_o_metrics = extract_metrics_for_all_tasks(full_model, train_info["sessions"], device, train_info, 'token_micro_f1_no_o')
        
        logger.info(f"All historical tasks TEST metrics (acc): {all_acc_metrics}")
        logger.info(f"All historical tasks TEST metrics (chunk_f1): {all_chunk_f1_metrics}")
        logger.info(f"All historical tasks TEST metrics (token_micro_f1_no_o): {all_token_micro_f1_no_o_metrics}")
        
        # 添加当前任务的指标
        performance_list = all_acc_metrics + [current_test_metrics["acc"]]
        chunk_f1_list = all_chunk_f1_metrics + [current_test_metrics.get("chunk_f1", current_test_metrics["acc"])]
        token_micro_f1_no_o_list = all_token_micro_f1_no_o_metrics + [current_test_metrics.get("token_micro_f1_no_o", current_test_metrics["acc"])]
        
        logger.info(f"Current task TEST metrics - acc: {current_test_metrics['acc']:.4f}, "
                   f"chunk_f1: {current_test_metrics.get('chunk_f1', current_test_metrics['acc']):.4f}, "
                   f"token_micro_f1_no_o: {current_test_metrics.get('token_micro_f1_no_o', current_test_metrics['acc']):.4f}")
        logger.info(f"Final performance list (acc): {performance_list}")
        logger.info(f"Final performance list (chunk_f1): {chunk_f1_list}")
        logger.info(f"Final performance list (token_micro_f1_no_o): {token_micro_f1_no_o_list}")
    else:
        # 第一个任务，只有当前任务的准确率（使用TEST集指标）
        performance_list = [current_test_metrics["acc"]]
        chunk_f1_list = [current_test_metrics.get("chunk_f1", current_test_metrics["acc"])]
        token_micro_f1_no_o_list = [current_test_metrics.get("token_micro_f1_no_o", current_test_metrics["acc"])]
        
        logger.info(f"First task performance - acc: {performance_list[0]:.4f}, "
                   f"chunk_f1: {chunk_f1_list[0]:.4f}, "
                   f"token_micro_f1_no_o: {token_micro_f1_no_o_list[0]:.4f}")
    
    # ✨ 处理0样本指标（分别提取三种指标）
    zero_shot_chunk_f1_metrics = {}
    zero_shot_token_micro_f1_no_o_metrics = {}
    if zero_shot_metrics:
        for session_name, metrics in zero_shot_metrics.items():
            if metrics:
                zero_shot_chunk_f1_metrics[session_name] = {'chunk_f1': metrics.get('chunk_f1', metrics.get('acc', 0.0))}
                zero_shot_token_micro_f1_no_o_metrics[session_name] = {'token_micro_f1_no_o': metrics.get('token_micro_f1_no_o', metrics.get('acc', 0.0))}
    
    # ✨ 将三种指标传递给准确率矩阵
    cm.update_acc_matrix(
        task_idx, 
        performance_list, 
        zero_shot_metrics,
        chunk_f1_list,
        token_micro_f1_no_o_list,
        zero_shot_chunk_f1_metrics,
        zero_shot_token_micro_f1_no_o_metrics
    )
    train_info["acc_matrix"] = cm.acc_matrix
    train_info["chunk_f1_matrix"] = cm.chunk_f1_matrix
    train_info["token_micro_f1_no_o_matrix"] = cm.token_micro_f1_no_o_matrix
    
    # 添加当前任务到训练信息中
    train_info["sessions"].append(session_info)
    train_info["tasks"].append(args.task_name)
    
    # ========== 12) 计算持续学习指标 ==========
    # 若是第一个任务, 不算持续学习指标
    if len(train_info["sessions"]) <= 1:
        logger.info("[Info] This is the first task, skip any CL metrics.")
        final_metrics = {}
        final_metrics_chunk_f1 = {}
        final_metrics_token_micro_f1_no_o = {}
    else:
        k = len(train_info["sessions"])  # 总任务数
        from continual.metrics import compute_multimodal_transfer_metrics, analyze_task_similarity_transfer
        
        # 获取任务名称列表
        task_names = [session.get('task_name', 'unknown') for session in train_info["sessions"]]
        
        # ✨ 分别用三种指标计算持续学习指标
        logger.info("="*80)
        logger.info("📊 Computing Continual Learning Metrics with 3 different metrics:")
        logger.info("="*80)
        
        # 1. 默认指标（acc）
        logger.info(f"📈 Metric 1: Default (acc) - micro_f1 for sentence tasks, chunk_f1 for sequence tasks")
        final_metrics = compute_multimodal_transfer_metrics(cm, k, task_names, matrix_type='acc')
        similarity_analysis = analyze_task_similarity_transfer(cm, task_names, matrix_type='acc')
        if similarity_analysis:
            final_metrics.update(similarity_analysis)
        logger.info(f"  AA={final_metrics.get('AA', 0):.2f}, AIA={final_metrics.get('AIA', 0):.2f}, "
                   f"FM={final_metrics.get('FM', 0):.2f}, BWT={final_metrics.get('BWT', 0):.2f}")
        
        # 2. Chunk F1（仅对序列任务有效，句级任务回退到acc）
        logger.info(f"📈 Metric 2: Chunk-level F1 (for sequence tasks)")
        final_metrics_chunk_f1 = compute_multimodal_transfer_metrics(cm, k, task_names, matrix_type='chunk_f1')
        similarity_analysis_chunk = analyze_task_similarity_transfer(cm, task_names, matrix_type='chunk_f1')
        if similarity_analysis_chunk:
            final_metrics_chunk_f1.update(similarity_analysis_chunk)
        logger.info(f"  AA={final_metrics_chunk_f1.get('AA', 0):.2f}, AIA={final_metrics_chunk_f1.get('AIA', 0):.2f}, "
                   f"FM={final_metrics_chunk_f1.get('FM', 0):.2f}, BWT={final_metrics_chunk_f1.get('BWT', 0):.2f}")
        
        # 3. Token Micro F1 (no O)（仅对序列任务有效，句级任务回退到acc）
        logger.info(f"📈 Metric 3: Token-level Micro F1 (no O, for sequence tasks)")
        final_metrics_token_micro_f1_no_o = compute_multimodal_transfer_metrics(cm, k, task_names, matrix_type='token_micro_f1_no_o')
        similarity_analysis_token = analyze_task_similarity_transfer(cm, task_names, matrix_type='token_micro_f1_no_o')
        if similarity_analysis_token:
            final_metrics_token_micro_f1_no_o.update(similarity_analysis_token)
        logger.info(f"  AA={final_metrics_token_micro_f1_no_o.get('AA', 0):.2f}, AIA={final_metrics_token_micro_f1_no_o.get('AIA', 0):.2f}, "
                   f"FM={final_metrics_token_micro_f1_no_o.get('FM', 0):.2f}, BWT={final_metrics_token_micro_f1_no_o.get('BWT', 0):.2f}")
        
        logger.info("="*80)
    
    # ✨ 合并训练指标和持续学习指标（三种指标）
    session_info["final_metrics"] = {
        "best_metrics": train_result["best_metrics"],
        "continual_metrics": final_metrics,  # 默认指标（acc）
        "continual_metrics_chunk_f1": final_metrics_chunk_f1,  # Chunk F1
        "continual_metrics_token_micro_f1_no_o": final_metrics_token_micro_f1_no_o  # Token Micro F1 (no O)
    }
    
    # ========== 12.5) 特征聚类可视化 ==========
    if getattr(args, 'enable_feature_visualization', True):  # 默认开启可视化
        try:
            logger.info("="*60)
            logger.info("📊 开始特征聚类可视化...")
            logger.info("="*60)
            
            # 创建可视化保存目录
            vis_dir = os.path.join(os.path.dirname(args.output_model_path), 'feature_clustering')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 检查是否使用增强版可视化（真实vs预测对比）
            show_predictions = getattr(args, 'vis_show_predictions', True)  # 默认显示预测对比
            
            if show_predictions:
                # 使用增强版：生成真实标签图 + 预测对比图
                logger.info("📊 使用增强版可视化（包含预测对比图）")
                visualize_task_enhanced(
                    model=full_model,
                    task_name=args.task_name,
                    session_name=args.session_name,
                    device=device,
                    args=args,
                    save_dir=vis_dir,
                    split='dev',  # 使用验证集
                    max_samples=getattr(args, 'vis_max_samples', 2000),
                    show_predictions=True,  # 生成预测对比图
                    config_name=config_name,  # 传递配置文件名，避免覆盖
                    plot_dual_metrics=True  # ✨ 为序列任务生成两种指标的图
                )
            else:
                # 使用基础版：仅生成真实标签图
                logger.info("📊 使用基础版可视化（仅真实标签）")
                visualize_task_after_training(
                    model=full_model,
                    task_name=args.task_name,
                    session_name=args.session_name,
                    device=device,
                    args=args,
                    config_name=config_name,  # 传递配置文件名
                    save_dir=vis_dir,
                    split='dev',  # 使用验证集
                    max_samples=getattr(args, 'vis_max_samples', 2000),
                    use_both_methods=getattr(args, 'vis_use_both', False)
                )
            
            # 如果已经学习了多个任务，绘制演进图
            if len(train_info["sessions"]) >= 2:
                logger.info("📊 绘制持续学习演进图（所有已学习任务）...")
                visualize_all_tasks_evolution(
                    save_dir=vis_dir,
                    split='dev',
                    method='tsne',
                    config_name=config_name  # ✨ 传递config_name避免覆盖
                )
            
            logger.info("✓ 特征聚类可视化完成\n")
            
        except Exception as e:
            logger.warning(f"⚠️  特征可视化失败（不影响训练）: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # ========== 13) 保存模型 ==========
    logger.info(f"Saving model to: {args.output_model_path}")
    torch.save(full_model.state_dict(), args.output_model_path)
    
    # 保存任务头信息
    task_heads_path = args.output_model_path.replace('.pt', '_task_heads.pt')
    if hasattr(full_model, 'save_task_heads'):
        full_model.save_task_heads(task_heads_path)
        logger.info(f"Task heads saved to: {task_heads_path}")
    
    logger.info(f"Model saved successfully to: {args.output_model_path}")
    
    # ========== 15) 绘制训练曲线 ==========
    if getattr(args, 'plot_training_curves', True):  # 默认启用
        try:
            logger.info("="*80)
            logger.info("📈 绘制训练曲线...")
            logger.info("="*80)
            
            # 准备绘图数据
            epoch_losses = train_result.get("epoch_losses", [])
            dev_losses = train_result.get("dev_losses", [])
            dev_metrics_history = train_result.get("dev_metrics_history", [])
            
            if epoch_losses and dev_metrics_history:
                # 提取关键指标
                epochs = list(range(1, len(epoch_losses) + 1))
                # 如果dev_losses不存在或长度不匹配，用占位符
                if not dev_losses or len(dev_losses) != len(epoch_losses):
                    dev_losses = [0.0] * len(epochs)
                span_f1_scores = [m.get('acc', 0.0) for m in dev_metrics_history]  # 主指标
                
                metrics_history = {
                    'epochs': epochs,
                    'train_loss': epoch_losses,
                    'dev_loss': dev_losses,  # 验证loss（已在validate_epoch中计算）
                    'span_f1': span_f1_scores
                }
                
                # 确定保存路径
                curves_dir = os.path.dirname(args.output_model_path)
                curves_filename = f"{args.session_name}_training_curves.png"
                if config_name:
                    curves_filename = f"{config_name}_{args.session_name}_curves.png"
                curves_path = os.path.join(curves_dir, curves_filename)
                
                # 绘制曲线
                plot_training_curves(
                    metrics_history=metrics_history,
                    save_path=curves_path,
                    task_name=f"{args.task_name.upper()} ({args.session_name})",
                    figsize=(12, 6),
                    dpi=150
                )
                logger.info(f"✓ 训练曲线已保存: {curves_path}")
            else:
                logger.warning("⚠️ 训练历史数据不完整，跳过绘图")
        except Exception as e:
            logger.error(f"⚠️ 绘制训练曲线失败: {e}")
            import traceback
            traceback.print_exc()
    
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
