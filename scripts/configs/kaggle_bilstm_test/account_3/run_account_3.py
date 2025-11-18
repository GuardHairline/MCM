#!/usr/bin/env python3
"""
Kaggle运行脚本 - Account 3 - MABSA BiLSTM测试

此脚本在Kaggle Notebook中运行
任务: MABSA
配置: config_default
预计时间: 220 分钟

使用说明:
1. 在Kaggle Notebook中创建新的Code
2. 设置加速器为 GPU T4 或 P100
3. 添加数据集: mcm-project (包含代码和数据)
4. 复制此脚本内容到Notebook
5. 点击 Run All

测试目的:
- 验证新实现的BiLSTM-CRF任务头功能正确
- 比较不同BiLSTM超参数配置的效果
- 确认text_only → multimodal持续学习流程正常
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
import shutil

# ============================================================================
# Kaggle环境配置
# ============================================================================

KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"
PROJECT_DATASET = "mcm-project"  # 你的Kaggle数据集名称

print("="*80)
print("Account 3 - MABSA BiLSTM测试")
print("="*80)
print(f"任务: MABSA")
print(f"配置数: 1")
print(f"预计时间: 220 分钟")
print("="*80 + "\n")

# ============================================================================
# Step 1: 项目设置
# ============================================================================

def setup_project():
    """设置项目环境"""
    print("\n" + "="*80)
    print("Step 1: 设置项目环境")
    print("="*80)
    
    project_src = Path(KAGGLE_INPUT) / PROJECT_DATASET
    project_dst = Path(KAGGLE_WORKING) / "MCM"
    
    # 检查源目录
    if not project_src.exists():
        print(f"❌ 错误: 未找到项目数据集 {project_src}")
        print("   请确保已添加 mcm-project 数据集")
        sys.exit(1)
    
    # 复制项目文件
    if project_dst.exists():
        print(f"项目目录已存在，删除旧版本...")
        shutil.rmtree(project_dst)
    
    print(f"复制项目文件...")
    shutil.copytree(project_src, project_dst)
    
    # 切换到项目目录
    os.chdir(project_dst)
    sys.path.insert(0, str(project_dst))
    
    print(f"✓ 项目目录: {project_dst}")
    print(f"✓ 当前工作目录: {os.getcwd()}")
    
    return project_dst

# ============================================================================
# Step 2: 运行实验
# ============================================================================

def run_experiments(project_dir: Path):
    """运行所有实验"""
    print("\n" + "="*80)
    print("Step 2: 运行BiLSTM测试实验")
    print("="*80)
    
    configs = [{"file": "kaggle_bilstm_config_default_twitter2015_mabsa.json", "task": "mabsa", "config_type": "config_default"}]
    
    results = {}
    
    for idx, config_info in enumerate(configs, 1):
        config_file = config_info["file"]
        config_path = project_dir / "scripts" / "configs" / "kaggle_bilstm_test" / "account_3" / config_file
        
        print(f"\n{'='*80}")
        print(f"实验 {idx}/{len(configs)}: {config_info['config_type']}")
        print(f"{'='*80}")
        print(f"任务: {config_info['task'].upper()}")
        print(f"配置文件: {config_file}")
        
        start_time = time.time()
        
        try:
            # 运行训练
            cmd = [
                sys.executable, "-m", "scripts.train_with_zero_shot",
                "--config", str(config_path)
            ]
            
            print(f"\n执行命令: {' '.join(cmd)}\n")
            
            result = subprocess.run(
                cmd,
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=7200  # 2小时超时
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"\n✓ 实验完成，耗时: {elapsed/60:.1f} 分钟")
                results[config_file] = {
                    "status": "success",
                    "elapsed_minutes": elapsed / 60,
                    "config_type": config_info["config_type"]
                }
            else:
                print(f"\n❌ 实验失败")
                print("STDERR:", result.stderr[-500:] if result.stderr else "")
                results[config_file] = {
                    "status": "failed",
                    "error": result.stderr[-200:] if result.stderr else "Unknown error"
                }
        
        except subprocess.TimeoutExpired:
            print(f"\n❌ 实验超时（2小时）")
            results[config_file] = {
                "status": "timeout"
            }
        except Exception as e:
            print(f"\n❌ 实验异常: {e}")
            results[config_file] = {
                "status": "error",
                "error": str(e)
            }
    
    return results

# ============================================================================
# Step 3: 收集和导出结果
# ============================================================================

def collect_results(project_dir: Path, results: dict):
    """收集结果文件"""
    print("\n" + "="*80)
    print("Step 3: 收集结果")
    print("="*80)
    
    # 创建结果目录
    results_dir = Path(KAGGLE_WORKING) / "results_account_3"
    results_dir.mkdir(exist_ok=True)
    
    # 复制训练信息文件
    checkpoint_dir = project_dir / "checkpoints"
    if checkpoint_dir.exists():
        for train_info_file in checkpoint_dir.glob("train_info_*.json"):
            shutil.copy(train_info_file, results_dir)
            print(f"✓ {train_info_file.name}")
        
        # 复制图片文件
        for img_file in checkpoint_dir.glob("*.png"):
            shutil.copy(img_file, results_dir)
            print(f"✓ {img_file.name}")
    
    # 保存运行摘要
    summary_file = results_dir / "run_summary_account_3.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "account_id": "account_3",
            "task": "mabsa",
            "total_configs": len(results),
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"\n✓ 结果已保存到: {results_dir}")
    print(f"✓ 运行摘要: {summary_file}")
    
    return results_dir

# ============================================================================
# 主流程
# ============================================================================

if __name__ == "__main__":
    try:
        # Step 1: 设置项目
        project_dir = setup_project()
        
        # Step 2: 运行实验
        results = run_experiments(project_dir)
        
        # Step 3: 收集结果
        results_dir = collect_results(project_dir, results)
        
        # 打印最终摘要
        print("\n" + "="*80)
        print("✅ 所有实验完成")
        print("="*80)
        success_count = sum(1 for r in results.values() if r.get("status") == "success")
        print(f"成功: {success_count}/{len(results)}")
        print(f"结果目录: {results_dir}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
