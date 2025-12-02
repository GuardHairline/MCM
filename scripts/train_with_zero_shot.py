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
from modules.parser import create_train_parser, validate_args
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
        save_checkpoints = global_params.get("save_checkpoints", False)
        if save_checkpoints:
            print("="*60)
            print("🧹 清理已跳过：save_checkpoints=1，保留所有模型文件")
            print("="*60 + "\n")
            return

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
        
        # 需要处理的文件模式
        patterns_to_handle = [
            f"{base_name}.pt",                      # 主模型文件
            f"{base_name}_*.pt",                    # 其他相关模型文件
            f"model_{base_name}*.pt",               # 带model前缀的文件
            f"*{base_name}_task_heads.pt",          # 任务头文件
            f"label_embedding_{base_name}.pt",      # 标签嵌入文件
        ]
        
        processed_count = 0
        
        # 在checkpoint_dir中查找并处理匹配的文件
        for pattern in patterns_to_handle:
            full_pattern = os.path.join(checkpoint_dir, pattern)
            matching_files = glob.glob(full_pattern)
            
            for file_path in matching_files:
                file_name = os.path.basename(file_path)
                # 确保base_name在文件名中（额外安全检查）
                if base_name in file_name:
                    try:
                        os.remove(file_path)
                        print(f"  ✓ 删除: {file_name}")
                        processed_count += 1
                    except Exception as del_err:
                        print(f"  ✗ 删除失败: {file_name} ({del_err})")
        
        # 清理EWC参数
        ewc_dir = global_params.get("ewc_dir", "")
        if ewc_dir and os.path.exists(ewc_dir):
            ewc_pattern = os.path.join(ewc_dir, f"*{base_name}*.pt")
            for file_path in glob.glob(ewc_pattern):
                if base_name in os.path.basename(file_path):
                    try:
                        os.remove(file_path)
                        print(f"  ✓ 删除EWC: {os.path.basename(file_path)}")
                        processed_count += 1
                    except Exception as del_err:
                        print(f"  ✗ 删除EWC失败: {os.path.basename(file_path)} ({del_err})")
        
        # 清理GEM记忆
        gem_dir = global_params.get("gem_mem_dir", "")
        if gem_dir and os.path.exists(gem_dir):
            gem_pattern = os.path.join(gem_dir, f"*{base_name}*.pt")
            for file_path in glob.glob(gem_pattern):
                if base_name in os.path.basename(file_path):
                    try:
                        os.remove(file_path)
                        print(f"  ✓ 删除GEM: {os.path.basename(file_path)}")
                        processed_count += 1
                    except Exception as del_err:
                        print(f"  ✗ 删除GEM失败: {os.path.basename(file_path)} ({del_err})")
        
        print(f"\n✅ 清理完成: 删除 {processed_count} 个文件")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ 清理过程出错: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")


def _build_args(task_config: Dict[str, Any], global_params: Dict[str, Any], pretrained_model_path: str) -> argparse.Namespace:
    """
    依据 parser 定义构造完整 args：parser默认 -> global_params -> task_config
    """
    parser = create_train_parser()
    defaults = {action.dest: action.default for action in parser._actions if action.dest != "help"}
    args_dict = defaults.copy()

    def update_from(source: Dict[str, Any]):
        for k, v in source.items():
            if k in args_dict:
                args_dict[k] = v

    # 全局覆盖默认，再由任务覆盖
    update_from(global_params)
    update_from(task_config)

    # 必填/特殊字段
    args_dict["task_name"] = task_config["task_name"]
    args_dict["session_name"] = task_config["session_name"]
    args_dict["task_config_file"] = global_params.get("task_config_file", "")
    args_dict["train_info_json"] = global_params["train_info_json"]
    args_dict["output_model_path"] = task_config.get("output_model_path", global_params.get("output_model_path"))
    args_dict["pretrained_model_path"] = task_config.get("pretrained_model_path", pretrained_model_path)
    args_dict["data_dir"] = global_params.get("data_dir", args_dict.get("data_dir"))
    args_dict["dataset_name"] = global_params.get("dataset_name", args_dict.get("dataset_name"))
    args_dict["num_workers"] = global_params.get("num_workers", args_dict.get("num_workers", 4))

    # 兼容目录/描述字段
    if "gem_mem_dir" in global_params:
        args_dict["gem_mem_dir"] = global_params["gem_mem_dir"]
    if "ewc_dir" in global_params:
        args_dict["ewc_dir"] = global_params["ewc_dir"]
    if "description_file" in task_config:
        args_dict["description_file"] = task_config["description_file"]

    args = argparse.Namespace(**args_dict)
    validate_args(args)
    return args


def run_single_task(task_config: Dict[str, Any], global_params: Dict[str, Any], 
                   task_idx: int, total_tasks: int, pretrained_model_path: str = "", all_tasks: List[Dict[str, Any]] = []) -> str:
    """运行单个任务"""
    
    print(f"Running task {task_idx + 1}/{total_tasks}: {task_config['task_name']} ({task_config['session_name']})")
    
    # 构造完整参数
    args = _build_args(task_config, global_params, pretrained_model_path)
    
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
        return args.output_model_path
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
    
    global_params["kaggle_mode"] = config.get("kaggle_mode", global_params.get("kaggle_mode", False))
    
    # 确定任务范围
    start_idx = args.start_task
    end_idx = args.end_task if args.end_task is not None else len(tasks)
    # 确保end_idx不超过实际任务数量
    end_idx = min(end_idx, len(tasks))
    
    print(f"Total tasks: {len(tasks)}")
    print(f"Running tasks: {start_idx + 1} to {end_idx} (requested: {args.end_task})")
    
    # 打印配置信息（兼容不同配置格式）
    if "env" in config:
        print(f"Environment: {config['env']}")
    if "strategy" in config:
        print(f"Strategy: {config['strategy']}")
    if "mode_suffix" in config:
        print(f"Mode: {config['mode_suffix']}")
    if "dataset" in config:
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
    
    print("\n" + "="*80)
    print("✅ 所有任务训练完成！")
    print("="*80)
    print(f"Final model: {pretrained_model_path}")
    print(f"Training info: {global_params['train_info_json']}")
    
    # ========== 自动绘制热力图 ==========
    try:
        print("\n" + "="*80)
        print("📊 自动绘制持续学习热力图")
        print("="*80)
        
        from utils.plot import plot_accuracy_matrix_from_train_info
        import json
        import os
        
        train_info_path = global_params['train_info_json']
        output_dir = os.path.dirname(train_info_path)
        
        if os.path.exists(train_info_path):
            # 读取train_info
            with open(train_info_path, 'r', encoding='utf-8') as f:
                train_info = json.load(f)
            
            # 从train_info文件名提取配置ID（避免不同配置的图片互相覆盖）
            train_info_basename = os.path.basename(train_info_path)  # e.g., train_info_kaggle_mate_twitter2015_config_default.json
            config_id = train_info_basename.replace('train_info_', '').replace('.json', '')  # e.g., kaggle_mate_twitter2015_config_default
            
            # 绘制所有三种指标的热力图
            print("\n1. 绘制 Accuracy (Acc) 热力图...")
            if 'acc_matrix' in train_info and train_info['acc_matrix']:
                acc_save_path = os.path.join(output_dir, f'accuracy_heatmap_{config_id}.png')
                plot_accuracy_matrix_from_train_info(
                    train_info_path=train_info_path,
                    output_path=acc_save_path,
                    show_values=True,
                    metric='acc'
                )
                print(f"   ✓ Accuracy热力图: {acc_save_path}")
            else:
                print("   ⚠️ acc_matrix 不存在或为空")
            
            # 绘制 Chunk F1 热力图
            print("\n2. 绘制 Chunk F1 (Span F1) 热力图...")
            if 'chunk_f1_matrix' in train_info and train_info['chunk_f1_matrix']:
                chunk_f1_save_path = os.path.join(output_dir, f'chunk_f1_heatmap_{config_id}.png')
                plot_accuracy_matrix_from_train_info(
                    train_info_path=train_info_path,
                    output_path=chunk_f1_save_path,
                    show_values=True,
                    metric='chunk_f1'
                )
                print(f"   ✓ Chunk F1热力图: {chunk_f1_save_path}")
            else:
                print("   ⚠️ chunk_f1_matrix 不存在或为空")
            
            # 绘制 Token Micro F1 热力图
            print("\n3. 绘制 Token Micro F1 (no O) 热力图...")
            if 'token_micro_f1_no_o_matrix' in train_info and train_info['token_micro_f1_no_o_matrix']:
                token_f1_save_path = os.path.join(output_dir, f'token_micro_f1_heatmap_{config_id}.png')
                plot_accuracy_matrix_from_train_info(
                    train_info_path=train_info_path,
                    output_path=token_f1_save_path,
                    show_values=True,
                    metric='token_micro_f1_no_o'
                )
                print(f"   ✓ Token Micro F1热力图: {token_f1_save_path}")
            else:
                print("   ⚠️ token_micro_f1_no_o_matrix 不存在或为空")
            
            print("\n✅ 热力图绘制完成！")
        else:
            print(f"⚠️ train_info文件不存在: {train_info_path}")
            
    except Exception as e:
        print(f"⚠️ 绘制热力图失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 清理实验文件 ==========
    cleanup_experiment_files(config, global_params)


if __name__ == "__main__":
    main() 
