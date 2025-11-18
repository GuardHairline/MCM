#!/usr/bin/env python3
"""
Kaggle运行脚本 - Account 2 - MNER (Baseline vs CRF)

此脚本在Kaggle Notebook中运行
任务: MNER
配置: baseline, crf_only
预计时间: 400 分钟

使用说明:
1. 在Kaggle Notebook中创建新的Code
2. 设置加速器为 GPU P100
3. 添加数据集: mcm-project (包含代码和数据)
4. 复制此脚本内容到Notebook
5. 点击 Run All
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
print("Account 2 - MNER (Baseline vs CRF)")
print("="*80)
print(f"任务: MNER")
print(f"配置数: 2")
print(f"预计时间: 400 分钟")
print("="*80 + "\n")

# ============================================================================
# Step 1: 项目设置
# ============================================================================

print("\n" + "="*80)
print("Step 1: 设置项目")
print("="*80)

# 检查数据集
dataset_path = Path(KAGGLE_INPUT) / PROJECT_DATASET
if not dataset_path.exists():
    print(f"❌ 数据集未找到: {dataset_path}")
    print("请在Notebook设置中添加 '{PROJECT_DATASET}' 数据集")
    sys.exit(1)

print(f"✓ 数据集路径: {dataset_path}")

# 复制项目到工作目录
project_dir = Path(KAGGLE_WORKING) / "MCM"
if project_dir.exists():
    print(f"清理旧项目: {project_dir}")
    shutil.rmtree(project_dir)

print(f"复制项目到: {project_dir}")
shutil.copytree(dataset_path, project_dir)

# 切换到项目目录
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

print(f"✓ 当前工作目录: {os.getcwd()}")
print(f"✓ Python路径已更新")

# ============================================================================
# Step 2: 检查依赖
# ============================================================================

print("\n" + "="*80)
print("Step 2: 检查依赖")
print("="*80)

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("❌ PyTorch未安装")
    sys.exit(1)

# 安装pytorch-crf (如果需要)
try:
    from torchcrf import CRF
    print("✓ torchcrf已安装")
except ImportError:
    print("安装torchcrf...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pytorch-crf", "-q"], check=True)
    print("✓ torchcrf安装完成")

# ============================================================================
# Step 3: 运行实验
# ============================================================================

print("\n" + "="*80)
print("Step 3: 运行实验")
print("="*80)

# 配置文件列表
configs = [
  {
    "file": "kaggle_baseline_twitter2015_mner.json",
    "ablation": "baseline"
  },
  {
    "file": "kaggle_crf_only_twitter2015_mner.json",
    "ablation": "crf_only"
  }
]

results = []
start_time = time.time()

for i, config_info in enumerate(configs, 1):
    config_file = Path("scripts/configs/kaggle_ablation/account_2") / config_info["file"]
    ablation_type = config_info["ablation"]
    
    print(f"\n{'-'*80}")
    print(f"实验 {i}/{len(configs)}: {ablation_type}")
    print(f"配置: {config_file}")
    print(f"{'-'*80}")
    
    exp_start = time.time()
    
    try:
        # 运行训练
        cmd = [
            sys.executable, "-m", "scripts.train_with_zero_shot",
            "--config", str(config_file)
        ]
        
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            env={**os.environ, "PYTHONPATH": str(project_dir)},
            capture_output=True,
            text=True
        )
        
        exp_time = time.time() - exp_start
        
        if result.returncode == 0:
            print(f"✅ 实验 {i} 完成 ({exp_time/60:.1f} 分钟)")
            status = "success"
        else:
            print(f"❌ 实验 {i} 失败")
            print(f"错误: {result.stderr[-500:]}")
            status = "failed"
        
        results.append({
            "experiment_id": i,
            "ablation_type": ablation_type,
            "status": status,
            "time_minutes": exp_time / 60,
            "config_file": str(config_file)
        })
        
    except Exception as e:
        print(f"❌ 实验 {i} 异常: {e}")
        results.append({
            "experiment_id": i,
            "ablation_type": ablation_type,
            "status": "error",
            "error": str(e)
        })
    
    # 保存中间结果
    with open(Path(KAGGLE_WORKING) / "account_2_results.json", 'w') as f:
        json.dump(results, f, indent=2)

# ============================================================================
# Step 4: 分析结果
# ============================================================================

print("\n" + "="*80)
print("Step 4: 分析结果")
print("="*80)

# 读取所有生成的metrics JSON文件
metrics_dir = Path(KAGGLE_WORKING) / "checkpoints"
all_metrics = []

if metrics_dir.exists():
    for json_file in metrics_dir.rglob("*_metrics.json"):
        try:
            with open(json_file, 'r') as f:
                metrics = json.load(f)
                # 从文件名推断ablation类型
                file_name = json_file.stem
                if "baseline" in str(json_file):
                    ablation = "baseline"
                elif "crf_only" in str(json_file):
                    ablation = "crf_only"
                else:
                    ablation = "unknown"
                
                metrics["ablation_type"] = ablation
                metrics["file_path"] = str(json_file)
                all_metrics.append(metrics)
        except Exception as e:
            print(f"⚠️ 无法读取 {json_file}: {e}")

print(f"✓ 找到 {len(all_metrics)} 个结果文件")

# 生成对比分析
if len(all_metrics) >= 2:
    print("\n" + "-"*80)
    print("结果对比:")
    print("-"*80)
    
    # 按ablation类型分组
    baseline_results = [m for m in all_metrics if m["ablation_type"] == "baseline"]
    crf_results = [m for m in all_metrics if m["ablation_type"] == "crf_only"]
    
    if baseline_results and crf_results:
        # 取最新的结果（如果有多个）
        baseline = baseline_results[-1]
        crf = crf_results[-1]
        
        print(f"\n{assignment['task'].upper()} 任务:")
        print(f"  Baseline:")
        print(f"    - Token Acc: {baseline.get('token_accuracy', 'N/A'):.4f if isinstance(baseline.get('token_accuracy'), (int, float)) else 'N/A'}")
        print(f"    - Chunk F1:  {baseline.get('chunk_f1', 'N/A'):.4f if isinstance(baseline.get('chunk_f1'), (int, float)) else 'N/A'}")
        
        print(f"  CRF Only:")
        print(f"    - Token Acc: {crf.get('token_accuracy', 'N/A'):.4f if isinstance(crf.get('token_accuracy'), (int, float)) else 'N/A'}")
        print(f"    - Chunk F1:  {crf.get('chunk_f1', 'N/A'):.4f if isinstance(crf.get('chunk_f1'), (int, float)) else 'N/A'}")
        
        # 计算提升
        if isinstance(baseline.get('chunk_f1'), (int, float)) and isinstance(crf.get('chunk_f1'), (int, float)):
            improvement = crf['chunk_f1'] - baseline['chunk_f1']
            improvement_pct = (improvement / baseline['chunk_f1']) * 100 if baseline['chunk_f1'] > 0 else 0
            print(f"  Improvement:")
            print(f"    - Chunk F1: +{improvement:.4f} ({improvement_pct:+.1f}%)")
else:
    print("⚠️ 结果不足，无法生成对比分析")

# ============================================================================
# Step 5: 保存结果
# ============================================================================

total_time = time.time() - start_time

print("\n" + "="*80)
print("Step 5: 保存结果")
print("="*80)

# 保存执行摘要
final_results = {
    "account_id": "account_2",
    "account_name": "Account 2 - MNER (Baseline vs CRF)",
    "task": "mner",
    "ablations": ['baseline', 'crf_only'],
    "total_experiments": len(results),
    "successful": sum(1 for r in results if r['status'] == 'success'),
    "failed": sum(1 for r in results if r['status'] in ['failed', 'error']),
    "total_time_minutes": total_time / 60,
    "experiments": results,
    "metrics": all_metrics
}

output_file = Path(KAGGLE_WORKING) / "account_2_final_results.json"
with open(output_file, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"✓ 结果已保存: {output_file}")

# 打包所有结果
print("\n打包结果文件...")
import zipfile

# 打包1: metrics和摘要JSON
zip_path = Path(KAGGLE_WORKING) / "account_2_results.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    # 添加摘要JSON
    zipf.write(output_file, arcname=output_file.name)
    
    # 添加所有metrics JSON
    for metrics in all_metrics:
        file_path = Path(metrics["file_path"])
        if file_path.exists():
            zipf.write(file_path, arcname=f"metrics/{file_path.name}")

print(f"✓ 结果已打包: {zip_path}")

# 打包2: 单独压缩所有模型文件（.pt）
print("\n压缩模型文件...")
models_zip_path = Path(KAGGLE_WORKING) / "account_2_models.zip"
pt_files_count = 0

with zipfile.ZipFile(models_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    # 遍历工作目录查找所有 .pt 文件
    for pt_file in Path(KAGGLE_WORKING).rglob("*.pt"):
        # 添加到压缩包，保留相对路径结构
        arcname = pt_file.relative_to(KAGGLE_WORKING)
        zipf.write(pt_file, arcname=str(arcname))
        pt_files_count += 1
        print(f"  压缩: {arcname}")

print(f"✓ 模型文件已压缩: {models_zip_path}")
print(f"  共 {pt_files_count} 个模型文件")

print("\n" + "="*80)
print("✅ 所有任务完成！")
print("="*80)
print(f"总实验数: {len(results)}")
print(f"成功: {sum(1 for r in results if r['status'] == 'success')}")
print(f"失败: {sum(1 for r in results if r['status'] in ['failed', 'error'])}")
print(f"总时间: {total_time/60:.1f} 分钟")
print("\n请下载以下文件:")
print(f"  1. {output_file.name} - JSON格式的详细结果")
print(f"  2. {zip_path.name} - 打包的指标和结果文件")
print(f"  3. {models_zip_path.name} - 打包的所有模型文件（.pt）")
print("="*80)
