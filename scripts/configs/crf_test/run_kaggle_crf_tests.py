#!/usr/bin/env python3
"""
Kaggle环境批量运行CRF测试
"""
import subprocess
import json
from pathlib import Path
import time
import os
import sys

def run_experiment(config_file):
    """运行单个实验"""
    print("="*80)
    print(f"Running: {config_file}")
    print("="*80)
    
    # 确保工作目录正确
    project_root = Path("/MCM")
    if not project_root.exists():
        project_root = Path.cwd()
    
    print(f"Project root: {project_root}")
    print(f"Config file: {config_file}")
    
    # 设置环境变量确保Python能找到模块
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    cmd = [
        "python", "-m", "scripts.train_with_zero_shot",
        "--config", config_file
    ]
    
    start_time = time.time()
    # 在项目根目录运行，并设置PYTHONPATH
    result = subprocess.run(cmd, capture_output=False, cwd=str(project_root), env=env)
    elapsed = time.time() - start_time
    
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"Exit code: {result.returncode}\n")
    
    return result.returncode == 0

def main():
    # 确保在正确的工作目录
    project_root = Path("/MCM")
    if project_root.exists():
        os.chdir(project_root)
        sys.path.insert(0, str(project_root))
        print(f"✓ 切换到项目目录: {project_root}")
    else:
        print("⚠️ 使用当前目录作为项目根目录")
    
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path[0]}\n")
    
    # 加载配置索引
    with open("scripts/configs/crf_test/test_index.json") as f:
        index = json.load(f)
    
    print("="*80)
    print("CRF修复测试 - Kaggle批量运行")
    print("="*80)
    print(f"\n总实验数: {index['total_configs']}")
    print(f"任务: {', '.join(index['tasks'])}")
    print("\n")
    
    results = []
    for i, config_info in enumerate(index['configs'], 1):
        config_file = f"scripts/configs/crf_test/{config_info['file']}"
        task = config_info['task']
        
        print(f"\n[{i}/{index['total_configs']}] {task.upper()}")
        success = run_experiment(config_file)
        
        results.append({
            "task": task,
            "config": config_file,
            "success": success
        })
        
        # 保存中间结果
        with open("/kaggle/working/test_progress.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # 打印总结
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)
    success_count = sum(1 for r in results if r['success'])
    print(f"\n成功: {success_count}/{len(results)}")
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"  {status} {r['task']}")
    
    print("\n结果保存在: /kaggle/working/")
    print("  - checkpoints/: 模型文件")
    print("  - train_info*.json: 训练详情")
    print("  - test_progress.json: 测试进度")

if __name__ == "__main__":
    main()
