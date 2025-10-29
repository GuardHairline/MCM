#!/usr/bin/env python3
"""
支持0样本检测的持续学习训练脚本

使用任务配置文件进行训练，可以在训练第i个任务时对第i+1、i+2等任务进行0样本检测。
"""

import json
import argparse
import sys
import torch
import torch.multiprocessing as mp
import glob
import os
from pathlib import Path
from typing import Dict, List, Any

# 设置文件系统共享策略，解决"Too many open files"问题
mp.set_sharing_strategy('file_system')

# 导入训练模块
from modules.train_refactored import train
from modules.parser import parse_train_args
from utils.logger import setup_logger


def load_task_config(config_file: str) -> Dict[str, Any]:
    """加载任务配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def cleanup_experiment_files(config: Dict[str, Any], global_params: Dict[str, Any]):
    """
    清理本次实验生成的.pt文件
    
    只删除当前实验的文件，不影响其他实验的文件
    根据配置文件名来识别相关文件
    
    Args:
        config: 配置字典
        global_params: 全局参数字典
    """
    try:
        print("="*60)
        print("🧹 开始清理实验文件...")
        print("="*60)
        
        # 从global_params中提取文件名模式
        # 例如: checkpoints/twitter2015_none_t2m_hp1.pt
        model_path = global_params.get("output_model_path", "")
        if not model_path:
            print("⚠️  未找到模型路径，跳过清理")
            return
        
        # 提取base_name: twitter2015_none_t2m_hp1
        model_file = Path(model_path)
        base_name = model_file.stem  # 不含.pt扩展名
        checkpoint_dir = model_file.parent
        
        print(f"📝 识别模式: {base_name}")
        print(f"📁 检查目录: {checkpoint_dir}")
        
        # 需要删除的文件模式
        patterns_to_delete = [
            f"{base_name}.pt",                      # 主模型文件
            f"{base_name}_*.pt",                    # 其他相关模型文件
            f"model_{base_name}*.pt",               # 带model前缀的文件
            f"*{base_name}_task_heads.pt",          # 任务头文件
            f"label_embedding_{base_name}.pt",      # 标签嵌入文件
        ]
        
        deleted_count = 0
        kept_count = 0
        
        # 在checkpoint_dir中查找并删除匹配的文件
        for pattern in patterns_to_delete:
            full_pattern = os.path.join(checkpoint_dir, pattern)
            matching_files = glob.glob(full_pattern)
            
            for file_path in matching_files:
                file_name = os.path.basename(file_path)
                # 确保base_name在文件名中（额外安全检查）
                if base_name in file_name:
                    try:
                        os.remove(file_path)
                        print(f"  ✓ 删除: {file_name}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  ✗ 删除失败: {file_name} ({e})")
                else:
                    kept_count += 1
        
        # 清理EWC参数
        ewc_dir = global_params.get("ewc_dir", "")
        if ewc_dir and os.path.exists(ewc_dir):
            ewc_pattern = os.path.join(ewc_dir, f"*{base_name}*.pt")
            for file_path in glob.glob(ewc_pattern):
                if base_name in os.path.basename(file_path):
                    try:
                        os.remove(file_path)
                        print(f"  ✓ 删除EWC: {os.path.basename(file_path)}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  ✗ 删除EWC失败: {os.path.basename(file_path)} ({e})")
        
        # 清理GEM记忆
        gem_dir = global_params.get("gem_mem_dir", "")
        if gem_dir and os.path.exists(gem_dir):
            gem_pattern = os.path.join(gem_dir, f"*{base_name}*.pt")
            for file_path in glob.glob(gem_pattern):
                if base_name in os.path.basename(file_path):
                    try:
                        os.remove(file_path)
                        print(f"  ✓ 删除GEM: {os.path.basename(file_path)}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  ✗ 删除GEM失败: {os.path.basename(file_path)} ({e})")
        
        print(f"\n✅ 清理完成: 删除 {deleted_count} 个文件")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ 清理过程出错: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")


def run_single_task(task_config: Dict[str, Any], global_params: Dict[str, Any], 
                   task_idx: int, total_tasks: int, pretrained_model_path: str = "", all_tasks: List[Dict[str, Any]] = []) -> str:
    """运行单个任务"""
    
    print(f"Running task {task_idx + 1}/{total_tasks}: {task_config['task_name']} ({task_config['session_name']})")
    
    # 创建参数对象
    args = argparse.Namespace()
    
    # 基本参数
    args.task_name = task_config["task_name"]
    args.session_name = task_config["session_name"]
    args.task_config_file = global_params.get("task_config_file", "")
    args.train_info_json = global_params["train_info_json"]
    args.output_model_path = global_params["output_model_path"]
    args.pretrained_model_path = pretrained_model_path
    
    # 数据参数
    args.data_dir = global_params.get("data_dir", "data")
    args.dataset_name = global_params.get("dataset_name", "twitter2015")
    args.train_text_file = task_config["train_text_file"]
    args.test_text_file = task_config["test_text_file"]
    args.dev_text_file = task_config["dev_text_file"]
    args.image_dir = task_config["image_dir"]
    
    # 模型参数
    args.text_model_name = task_config["text_model_name"]
    args.image_model_name = task_config["image_model_name"]
    args.fusion_strategy = task_config["fusion_strategy"]
    args.num_heads = task_config["num_heads"]
    args.mode = task_config["mode"]
    args.hidden_dim = task_config["hidden_dim"]
    args.dropout_prob = task_config["dropout_prob"]
    args.num_labels = task_config["num_labels"]
    
    # 训练参数
    args.epochs = task_config["epochs"]
    args.batch_size = task_config["batch_size"]
    args.lr = task_config["lr"]
    args.weight_decay = task_config["weight_decay"]
    args.step_size = task_config["step_size"]
    args.gamma = task_config["gamma"]
    args.patience = task_config["patience"]
    args.num_workers = global_params.get("num_workers", 4)
    
    # 持续学习策略参数
    for key in ["ewc", "ewc_lambda", "replay", "memory_percentage", "replay_ratio", 
                "replay_frequency", "lwf", "lwf_T", "lwf_alpha", "lwf_decay",
                "si", "si_epsilon", "si_decay", "mas", "mas_eps", "gem", "gem_mem",
                "pnn", "tam_cl", "moe_adapters", "moe_num_experts", "moe_top_k",
                "ddas", "ddas_threshold", "clap4clip", "mymethod", "deqa"]:
        if key in task_config:
            setattr(args, key, task_config[key])
        else:
            setattr(args, key, 0)  # 默认关闭
    
    # DEQA特定参数
    if task_config.get("deqa", 0):
        args.deqa_use_description = task_config.get("deqa_use_description", True)
        args.deqa_use_clip = task_config.get("deqa_use_clip", True)
        args.deqa_ensemble_method = task_config.get("deqa_ensemble_method", "weighted")
        args.deqa_freeze_old_experts = task_config.get("deqa_freeze_old_experts", True)
        args.deqa_distill_weight = task_config.get("deqa_distill_weight", 0.5)
        args.description_file = task_config.get("description_file", None)
    
    # 标签嵌入参数
    args.use_label_embedding = task_config.get("use_label_embedding", False)
    args.label_emb_dim = task_config.get("label_emb_dim", 128)
    args.use_similarity_reg = task_config.get("use_similarity_reg", True)
    args.similarity_weight = task_config.get("similarity_weight", 0.1)
    args.label_embedding_path = task_config.get("label_embedding_path", None)
    
    # 模型头部参数
    args.triaffine = task_config.get("triaffine", 1)
    args.span_hidden = task_config.get("span_hidden", 256)
    
    # 图平滑参数
    args.graph_smooth = task_config.get("graph_smooth", 1)
    args.graph_tau = task_config.get("graph_tau", 0.5)
    
    # 目录参数
    args.ewc_dir = global_params["ewc_dir"]
    args.gem_mem_dir = global_params["gem_mem_dir"]
    
    # 日志参数
    args.log_file = None
    
    print(f"Task parameters:")
    print(f"  Task: {args.task_name}")
    print(f"  Session: {args.session_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Strategy: {task_config.get('strategy', 'none')}")
    if args.use_label_embedding:
        print(f"  Label embedding: enabled")
    
    # 设置日志
    logger = setup_logger(args=args)
    
    try:
        # 直接调用训练函数
        best_metrics = train(args, logger, all_tasks=all_tasks)
        print(f"Task {task_idx + 1} completed successfully")
        print(f"Best metrics: {best_metrics}")
        return global_params["output_model_path"]
    except Exception as e:
        print(f"Task {task_idx + 1} failed with error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="持续学习训练脚本（支持0样本检测）")
    parser.add_argument("--config", type=str, required=True, default="scripts/task_config.json",
                       help="任务配置文件路径")
    parser.add_argument("--start_task", type=int, default=0,
                       help="开始任务索引（0-based）")
    parser.add_argument("--end_task", type=int, default=8,
                       help="结束任务索引（0-based，不包含）")
    
    args = parser.parse_args()
    
    # 加载任务配置
    print(f"Loading task configuration from: {args.config}")
    config = load_task_config(args.config)
    
    tasks = config["tasks"]
    global_params = config["global_params"]
    global_params["task_config_file"] = args.config  # 添加配置文件路径
    
    # 确定任务范围
    start_idx = args.start_task
    end_idx = args.end_task if args.end_task is not None else len(tasks)
    
    print(f"Total tasks: {len(tasks)}")
    print(f"Running tasks: {start_idx + 1} to {end_idx}")
    print(f"Environment: {config['env']}")
    print(f"Strategy: {config['strategy']}")
    print(f"Mode: {config['mode_suffix']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Label embedding: {'Yes' if config.get('use_label_embedding', False) else 'No'}")
    
    # 确保目录存在
    Path(global_params["train_info_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(global_params["output_model_path"]).parent.mkdir(parents=True, exist_ok=True)
    
    # 按顺序执行任务
    pretrained_model_path = ""
    for i in range(start_idx, end_idx):
        task_config = tasks[i]
        
        # 运行任务
        model_path = run_single_task(task_config, global_params, i, len(tasks), pretrained_model_path, all_tasks=tasks)
        
        # 更新预训练模型路径
        pretrained_model_path = model_path
        
        print(f"Completed task {i + 1}/{len(tasks)}: {task_config['task_name']}")
        print(f"Model saved to: {model_path}")
        print("-" * 50)
    
    print("All tasks completed successfully!")
    print(f"Final model: {pretrained_model_path}")
    print(f"Training info: {global_params['train_info_json']}")
    
    # ========== 自动绘制acc热力图 ==========
    from utils.plot import plot_acc_matrix_from_config
    plot_acc_matrix_from_config(
        config_file_path=args.config,
        train_info_file_path=global_params['train_info_json'],
        save_dir="checkpoints/acc_matrix"
    )
    
    # ========== 清理实验文件 ==========
    cleanup_experiment_files(config, global_params)


if __name__ == "__main__":
    main() 