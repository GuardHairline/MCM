#!/usr/bin/env python3
"""
Ablation Study结果分析脚本

分析6个账号的实验结果，生成对比报告
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_results(results_dir="results"):
    """加载所有账号的结果"""
    results_path = Path(results_dir)
    all_results = {}
    
    for i in range(1, 7):
        result_file = results_path / f"account_{i}_final_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                all_results[f"account_{i}"] = json.load(f)
            print(f"✓ Loaded: {result_file}")
        else:
            print(f"⚠️  Missing: {result_file}")
    
    return all_results

def extract_metrics(all_results):
    """提取所有指标"""
    data = []
    
    for account_id, account_data in all_results.items():
        task = account_data["task"]
        
        for exp in account_data["experiments"]:
            if exp["status"] == "success":
                # 这里需要根据实际输出提取metrics
                # 假设metrics保存在单独的文件中
                data.append({
                    "account": account_id,
                    "task": task,
                    "ablation": exp["ablation_type"],
                    "time_minutes": exp["time_minutes"],
                    # TODO: 添加从结果文件中读取的metrics
                })
    
    return pd.DataFrame(data)

def generate_report(df, output_dir="results"):
    """生成分析报告"""
    output_path = Path(output_dir)
    
    # 生成总结
    summary = {
        "total_experiments": len(df),
        "successful_experiments": len(df[df["time_minutes"].notna()]),
        "total_time_hours": df["time_minutes"].sum() / 60,
        "tasks": df["task"].unique().tolist(),
        "ablations": df["ablation"].unique().tolist()
    }
    
    with open(output_path / "ablation_study_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved: {output_path / 'ablation_study_summary.json'}")
    
    # 生成markdown报告
    with open(output_path / "ablation_study_report.md", 'w') as f:
        f.write("# Ablation Study Results\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Total Experiments: {summary['total_experiments']}\n")
        f.write(f"- Successful: {summary['successful_experiments']}\n")
        f.write(f"- Total Time: {summary['total_time_hours']:.1f} hours\n")
        f.write(f"\n## Results by Task\n\n")
        
        for task in summary['tasks']:
            task_df = df[df['task'] == task]
            f.write(f"### {task.upper()}\n\n")
            f.write(task_df.to_markdown(index=False))
            f.write("\n\n")
    
    print(f"✓ Report saved: {output_path / 'ablation_study_report.md'}")

def main():
    print("="*80)
    print("Ablation Study结果分析")
    print("="*80)
    
    # 加载结果
    all_results = load_all_results()
    
    if not all_results:
        print("\n❌ 没有找到结果文件")
        print("请确保将所有 account_X_final_results.json 放在 results/ 目录")
        return
    
    print(f"\n✓ 加载了 {len(all_results)} 个账号的结果\n")
    
    # 提取指标
    df = extract_metrics(all_results)
    
    # 生成报告
    generate_report(df)
    
    print("\n✅ 分析完成！")

if __name__ == "__main__":
    main()
