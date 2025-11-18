#!/usr/bin/env python3
"""
BiLSTM测试结果分析脚本

分析从Kaggle下载的结果文件，生成对比报告
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_dir: Path):
    """加载所有结果文件"""
    results = {}
    
    for train_info_file in results_dir.glob("train_info_*.json"):
        with open(train_info_file, 'r') as f:
            data = json.load(f)
            
            # 提取关键信息
            config_name = train_info_file.stem.replace("train_info_", "")
            results[config_name] = data
    
    return results

def extract_metrics(results: dict):
    """提取关键指标"""
    records = []
    
    for config_name, data in results.items():
        # 解析配置名称
        parts = config_name.split("_")
        task = parts[-1]
        config_type = parts[-2]
        
        for session in data.get("sessions", []):
            session_name = session.get("session_name", "")
            mode = "multimodal" if "multimodal" in session_name else "text_only"
            
            # 提取指标
            best_metric_summary = session.get("details", {}).get("best_metric_summary", {})
            final_test_metrics = session.get("details", {}).get("final_test_metrics", {})
            
            records.append({
                "task": task,
                "config_type": config_type,
                "mode": mode,
                "best_epoch": best_metric_summary.get("best_epoch", 0),
                "best_dev_metric": best_metric_summary.get("best_dev_metric", 0.0),
                "test_chunk_f1": final_test_metrics.get("chunk_f1", 0.0),
                "test_token_micro_f1": final_test_metrics.get("token_micro_f1", 0.0),
            })
    
    return pd.DataFrame(records)

def generate_report(df: pd.DataFrame, output_dir: Path):
    """生成分析报告"""
    report_path = output_dir / "bilstm_analysis_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# BiLSTM测试结果分析报告\n\n")
        
        # 按任务分组
        f.write("## 按任务分析\n\n")
        for task in df["task"].unique():
            task_df = df[df["task"] == task]
            f.write(f"### {task.upper()}\n\n")
            f.write(task_df.to_markdown(index=False))
            f.write("\n\n")
        
        # 按配置分组
        f.write("## 按配置分析\n\n")
        for config_type in df["config_type"].unique():
            config_df = df[df["config_type"] == config_type]
            f.write(f"### {config_type}\n\n")
            f.write(config_df.to_markdown(index=False))
            f.write("\n\n")
        
        # 最佳配置
        f.write("## 最佳配置推荐\n\n")
        best_by_task = df.loc[df.groupby("task")["test_chunk_f1"].idxmax()]
        f.write(best_by_task[["task", "config_type", "mode", "test_chunk_f1"]].to_markdown(index=False))
        f.write("\n")
    
    print(f"✓ 报告已生成: {report_path}")

def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """绘制对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, task in enumerate(df["task"].unique()):
        task_df = df[df["task"] == task]
        
        ax = axes[idx]
        sns.barplot(data=task_df, x="config_type", y="test_chunk_f1", hue="mode", ax=ax)
        ax.set_title(f"{task.upper()} - Chunk F1")
        ax.set_ylabel("Chunk F1 (%)")
        ax.set_xlabel("Configuration")
    
    plt.tight_layout()
    plot_path = output_dir / "bilstm_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✓ 对比图已生成: {plot_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python analyze_bilstm_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    
    print("加载结果...")
    results = load_results(results_dir)
    
    print("提取指标...")
    df = extract_metrics(results)
    
    print("生成报告...")
    generate_report(df, results_dir)
    
    print("绘制对比图...")
    plot_comparison(df, results_dir)
    
    print("\n✅ 分析完成")
