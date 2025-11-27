# modules/parser.py
import argparse
from typing import Any, Dict


def create_train_parser() -> argparse.ArgumentParser:
    """创建完整的训练参数解析器"""
    parser = argparse.ArgumentParser(description="Multi-task Continual Learning Training")
    
    # ========== 基本参数 ==========
    parser.add_argument("--task_name", type=str, required=True, 
                       choices=["mabsa", "masc", "mate", "mner"],
                       help="Name of the new task to train")
    parser.add_argument("--session_name", type=str, required=True,
                       help="Name or ID for this training session")
    parser.add_argument("--train_info_json", type=str, required=True,
                       help="Path to record train info (tasks, data, metrics, etc.)")
    parser.add_argument("--pretrained_model_path", type=str, default="",
                       help="Path to a pretrained model to continue training")
    parser.add_argument("--output_model_path", type=str, required=True,
                       help="Output model path")
    
    # ========== 数据参数 ==========
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Data directory")
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="Dataset name")
    parser.add_argument("--train_text_file", type=str, default="",
                       help="Train text file path")
    parser.add_argument("--test_text_file", type=str, default="",
                       help="Test text file path")
    parser.add_argument("--dev_text_file", type=str, default="",
                       help="Dev text file path")
    parser.add_argument("--image_dir", type=str, default="data/img",
                       help="Image directory")
    
    # ========== 模型参数 ==========
    parser.add_argument("--text_model_name", type=str, default="microsoft/deberta-v3-base",
                       help="Text model name")
    parser.add_argument("--image_model_name", type=str, default="google/vit-base-patch16-224-in21k",
                       help="Image model name")
    parser.add_argument("--fusion_strategy", type=str, default="concat",
                       choices=["concat", "multi_head_attention", "add"],
                       help="Multimodal fusion strategy")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--mode", type=str, default="multimodal", 
                       choices=["text_only", "multimodal"],
                       help="Model mode")
    parser.add_argument("--hidden_dim", type=int, default=768,
                       help="Hidden dimension")
    parser.add_argument("--dropout_prob", type=float, default=0.3,
                       help="Dropout probability in Full_Model")
    parser.add_argument("--num_labels", type=int, default=-1,
                       help="Number of labels (auto-detect if -1)")
    
    # BiLSTM-CRF参数
    parser.add_argument("--use_bilstm", type=int, default=1,
                       help="Use BiLSTM layer in task heads (1=True, 0=False, default=1)")
    parser.add_argument("--enable_bilstm_head", type=int, default=1,
                       help="Global switch for BiLSTM heads (1=enabled, 0=disable even if config requests)")
    parser.add_argument("--use_crf", type=int, default=1,
                       help="Use CRF layer for sequence labeling (1=True, 0=False, default=1)")
    parser.add_argument("--bilstm_hidden_size", type=int, default=256,
                       help="Hidden size of BiLSTM layer (default=256)")
    parser.add_argument("--bilstm_num_layers", type=int, default=2,
                       help="Number of BiLSTM layers (default=2)")
    
    # ========== 训练参数 ==========
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate for text encoder")
    parser.add_argument("--lstm_lr", type=float, default=1e-4,
                       help="Learning rate for BiLSTM layer (if use_bilstm=True)")
    parser.add_argument("--crf_lr", type=float, default=1e-3,
                       help="Learning rate for CRF layer (if use_bilstm=True and use_crf=True)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay (L2 regularization)")
    parser.add_argument("--step_size", type=int, default=0,
                       help="StepLR step size (<=0 to disable)")
    parser.add_argument("--gamma", type=float, default=0.5,
                       help="LR scheduler gamma for StepLR")
    parser.add_argument("--lr_scheduler", type=str, default="linear",
                       choices=["linear", "step", "none"],
                       help="Type of LR scheduler to use")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio for linear scheduler (0 to disable)")
    parser.add_argument("--patience", type=int, default=5,
                       help="Patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for data loading")
    parser.add_argument("--save_checkpoints", type=int, default=0,
                       help="Set to 1 to keep checkpoint files (skip cleanup step)")
    
    # ========== 标签嵌入参数 ==========
    parser.add_argument("--use_label_embedding", action="store_true",
                       help="Use label embedding")
    parser.add_argument("--use_hierarchical_head", action="store_true",
                       help="Use hierarchical multitask model with token and sentence heads")
    parser.add_argument("--label_emb_dim", type=int, default=128,
                       help="Label embedding dimension")
    parser.add_argument("--use_similarity_reg", action="store_true", default=False,
                       help="Use similarity regularization")
    parser.add_argument("--similarity_weight", type=float, default=0.1,
                       help="Similarity regularization weight")
    parser.add_argument("--label_embedding_path", type=str, default=None,
                       help="Label embedding save/load path")
    parser.add_argument("--label_emb_path", type=str, default="checkpoints/label_embedding.pt",
                       help="Path to label embedding (legacy)")
    parser.add_argument("--debug_samples", type=int, default=0,
                       help="Number of dev samples to dump gold/pred spans for debugging")
    
    # ========== 持续学习策略参数 ==========
    # EWC
    parser.add_argument("--ewc", type=int, default=0,
                       help="Whether to use EWC")
    parser.add_argument("--ewc_dir", type=str, default="ewc_params",
                       help="Directory to save EWC params")
    parser.add_argument("--ewc_lambda", type=float, default=0.01,
                       help="EWC lambda")
    parser.add_argument("--parallel", type=int, default=0,
                       help="Whether to use parallel EWC")
    
    # Experience Replay
    parser.add_argument("--replay", type=int, default=0,
                       help="Whether to use experience replay")
    parser.add_argument("--memory_percentage", type=float, default=0.05,
                       help="Memory percentage for replay")
    parser.add_argument("--replay_ratio", type=float, default=0.5,
                       help="Ratio of replay samples to total samples")
    parser.add_argument("--replay_frequency", type=int, default=4,
                       help="Replay frequency (epochs)")
    parser.add_argument("--memory_sampling_strategy", type=str, default='random',
                       choices=['random', 'random-balanced'],
                       help="Strategy for sampling memory buffer samples")
    
    # LwF
    parser.add_argument("--lwf", type=int, default=0,
                       help="Whether to use LwF")
    parser.add_argument("--lwf_T", type=float, default=2.0,
                       help="Temperature for LwF")
    parser.add_argument("--lwf_alpha", type=float, default=0.5,
                       help="Weight for LwF loss")
    parser.add_argument("--lwf_decay", type=float, default=0.5,
                       help="Decay rate for LwF_alpha")
    
    # SI
    parser.add_argument("--si", type=int, default=0,
                       help="Whether to use SI")
    parser.add_argument("--si_epsilon", type=float, default=0.1,
                       help="Epsilon for SI")
    parser.add_argument("--si_decay", type=float, default=0.5,
                       help="Decay rate for SI_epsilon")
    
    # MAS
    parser.add_argument("--mas", type=int, default=0,
                       help="Whether to use MAS")
    parser.add_argument("--mas_eps", type=float, default=1e-3,
                       help="Epsilon for MAS")
    parser.add_argument("--mas_decay", type=float, default=0.5,
                       help="Decay rate for MAS_eps")
    
    # GEM
    parser.add_argument("--gem", type=int, default=0,
                       help="Whether to use GEM")
    parser.add_argument("--gem_mem", type=int, default=100,
                       help="Memory size for GEM")
    parser.add_argument("--gem_dir", type=str, default="gem_memory",
                       help="Directory to save GEM memory")
    parser.add_argument("--gem_mem_dir", type=str, default="gem_memory",
                       help="Directory to save GEM memory (legacy)")
    
    # PNN
    parser.add_argument("--pnn", type=int, default=0,
                       help="Whether to use PNN")
    
    # TAM-CL
    parser.add_argument("--tam_cl", type=int, default=0,
                       help="Whether to use TAM-CL")
    parser.add_argument("--tam_alpha", type=float, default=1.0,
                    help="Weight α for intermediate KD loss in TAM-CL")
    
    # DEQA (Descriptions Enhanced Question-Answering Framework)
    parser.add_argument("--deqa", type=int, default=0,
                       help="Whether to use DEQA (Descriptions Enhanced QA Framework)")
    parser.add_argument("--description_file", type=str, default=None,
                       help="Path to image description file (.jsonl or .json)")
    parser.add_argument("--deqa_use_description", action="store_true", default=True,
                       help="Use description expert in DEQA")
    parser.add_argument("--deqa_use_clip", action="store_true", default=True,
                       help="Use CLIP expert in DEQA")
    parser.add_argument("--deqa_ensemble_method", type=str, default='weighted',
                       choices=['vote', 'weighted', 'learned'],
                       help="Ensemble method for DEQA experts")
    parser.add_argument("--deqa_freeze_old_experts", action="store_true", default=True,
                       help="Freeze old task experts in DEQA")
    parser.add_argument("--deqa_distill_weight", type=float, default=0.5,
                       help="Distillation loss weight for DEQA")

    
    # MoE Adapters
    parser.add_argument("--moe_adapters", type=int, default=0,
                       help="Enable MoE-Adapters baseline")
    parser.add_argument("--moe_num_experts", type=int, default=1,
                       help="Number of experts for MoE")
    parser.add_argument("--moe_top_k", type=int, default=1,
                       help="Top-k experts for MoE")
    parser.add_argument("--moe_expert_type", type=str, default="lora",
                    choices=["lora", "adapter"],
                    help="Expert type: LoRA or simple Adapter")
    parser.add_argument("--moe_balance_coef", type=float, default=0.01,
                       help="Coefficient for MoE load balancing loss")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="Rank of LoRA experts")
    parser.add_argument("--freeze_topk_experts", type=int, default=1,
                        help="Number of experts to freeze after each task")

    
    # DDAS
    parser.add_argument("--ddas", type=int, default=0,
                       help="Whether to use DDAS")
    parser.add_argument("--ddas_threshold", type=float, default=0.02,
                       help="Threshold for DDAS")
    
    # CLAP4CLIP
    parser.add_argument("--clap4clip", type=int, default=0,
                       help="Whether to use CL4CLAP")
    
    # MyMethod
    parser.add_argument("--mymethod", type=int, default=0,
                       help="Whether to use mymethod")
    
    # ========== 模型头部参数 ==========
    parser.add_argument("--triaffine", type=int, default=1,
                       help="Whether to use triaffine")
    parser.add_argument("--span_hidden", type=int, default=256,
                       help="Hidden dimension for span head")
    
    # ========== Span Loss参数 ==========
    parser.add_argument("--use_span_loss", type=int, default=1,
                       help="Whether to use span loss for sequence tasks (0=disable, 1=enable)")
    parser.add_argument("--boundary_weight", type=float, default=0.2,
                       help="Weight for boundary loss in span loss")
    parser.add_argument("--span_f1_weight", type=float, default=0.0,
                       help="Weight for span F1 loss (non-differentiable, usually 0)")
    parser.add_argument("--transition_weight", type=float, default=0.0,
                       help="Weight for transition penalty (non-differentiable, usually 0)")
    
    # ========== 图平滑参数 ==========
    parser.add_argument("--graph_smooth", type=int, default=0,
                       help="Whether to use label graph")
    parser.add_argument("--graph_tau", type=float, default=0.5,
                       help="Tau for label graph")
    
    # ========== 日志参数 ==========
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path")
    
    # ========== 任务配置文件参数（用于0样本检测） ==========
    parser.add_argument("--task_config_file", type=str, default=None,
                       help="Task configuration file for zero-shot evaluation")
    
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """验证参数的有效性"""
    # 检查任务名称
    valid_tasks = ["mabsa", "masc", "mate", "mner"]
    if args.task_name not in valid_tasks:
        raise ValueError(f"Invalid task_name: {args.task_name}. Must be one of {valid_tasks}")
    
    # 检查融合策略
    valid_fusion = ["concat", "multi_head_attention", "add"]
    if args.fusion_strategy not in valid_fusion:
        raise ValueError(f"Invalid fusion_strategy: {args.fusion_strategy}. Must be one of {valid_fusion}")
    
    # 检查模式
    valid_modes = ["text_only", "multimodal"]
    if args.mode not in valid_modes:
        raise ValueError(f"Invalid mode: {args.mode}. Must be one of {valid_modes}")
    
    # 检查持续学习策略冲突
    cl_methods = [args.ewc, args.replay, args.lwf, args.si, args.mas, args.gem, args.pnn, args.tam_cl, args.moe_adapters, args.clap4clip, args.mymethod, getattr(args, 'deqa', 0)]
    active_methods = [i for i, method in enumerate(cl_methods) if method]
    
    if len(active_methods) > 1:
        method_names = ["EWC", "Replay", "LwF", "SI", "MAS", "GEM", "PNN", "TAM-CL", "MoE-Adapters", "CLAP4CLIP", "MyMethod", "DEQA"]
        active_names = [method_names[i] for i in active_methods]
        print(f"Warning: Multiple continual learning methods active: {active_names}")
    
    # 检查标签嵌入参数
    if args.use_label_embedding and args.label_embedding_path is None:
        args.label_embedding_path = args.label_emb_path  # 使用legacy参数作为fallback


def get_default_args() -> Dict[str, Any]:
    """获取默认参数字典"""
    parser = create_train_parser()
    args = parser.parse_args([])  # 空列表使用默认值
    return vars(args)


def parse_train_args() -> argparse.Namespace:
    """解析训练参数并验证"""
    parser = create_train_parser()
    args = parser.parse_args()
    validate_args(args)
    return args 
