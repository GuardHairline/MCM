#!/usr/bin/env python3
"""
Kaggle超参数搜索结果分析脚本

从Kaggle下载的结果目录中提取和分析实验结果
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
import argparse


def calculate_metrics(acc_matrix):
    """计算持续学习指标"""
    n = len(acc_matrix)
    if n == 0:
        return {}
    
    # AA (Average Accuracy)
    aa = np.mean([acc_matrix[n-1][j] for j in range(n)])
    
    # AIA (Average Incremental Accuracy)
    aia_values = []
    for i in range(n):
        avg_acc = np.mean([acc_matrix[i][j] for j in range(i+1)])
        aia_values.append(avg_acc)
    aia = np.mean(aia_values)
    
    # FM (Forgetting Measure)
    fm_values = []
    for j in range(n-1):
        max_acc = max([acc_matrix[i][j] for i in range(j, n)])
        final_acc = acc_matrix[n-1][j]
        fm_values.append(max_acc - final_acc)
    fm = np.mean(fm_values) if fm_values else 0.0
    
    # BWT (Backward Transfer)
    bwt_values = []
    for j in range(n-1):
        final_acc = acc_matrix[n-1][j]
        acc_after_j = acc_matrix[j][j]
        bwt_values.append(final_acc - acc_after_j)
    bwt = np.mean(bwt_values) if bwt_values else 0.0
    
    # FWT (Forward Transfer)
    fwt_values = []
    for j in range(1, n):
        acc_before_j = acc_matrix[j-1][j] if j > 0 else 0.0
        fwt_values.append(acc_before_j)
    fwt = np.mean(fwt_values) if fwt_values else 0.0
    
    return {
        "AA": aa,
        "AIA": aia,
        "FM": fm,
        "BWT": bwt,
        "FWT": fwt
    }


def find_train_info_files(results_dir: Path):
    """查找所有train_info文件"""
    return list(results_dir.glob("**/train_info_*.json"))


def extract_hyperparams_from_filename(filename: str):
    """从文件名提取超参数信息"""
    # 文件名格式: train_info_twitter2015_none_t2m_hpX.json
    # 我们需要从配置文件或其他地方获取超参数
    return None


def analyze_results(results_dir: Path, output_dir: Path):
    """分析Kaggle结果"""
    
    print(f"分析目录: {results_dir}")
    print()
    
    # 查找所有train_info文件
    train_info_files = find_train_info_files(results_dir)
    
    if not train_info_files:
        print("❌ 未找到任何train_info文件")
        return
    
    print(f"找到 {len(train_info_files)} 个结果文件")
    print()
    
    all_results = []
    
    for train_info_path in train_info_files:
        print(f"处理: {train_info_path.name}", end=" ... ")
        
        try:
            with open(train_info_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            acc_matrix = np.array(data.get("accuracy_matrix", []))
            
            if len(acc_matrix) == 0:
                print("❌ 无准确率矩阵")
                continue
            
            # 计算指标
            metrics = calculate_metrics(acc_matrix)
            
            # 从文件名提取信息
            # 格式: train_info_<dataset>_<strategy>_<mode>_<seq>.json
            parts = train_info_path.stem.replace("train_info_", "").split("_")
            
            result = {
                "file": train_info_path.name,
                "dataset": parts[0] if len(parts) > 0 else "unknown",
                "strategy": parts[1] if len(parts) > 1 else "unknown",
                "mode": parts[2] if len(parts) > 2 else "unknown",
                "seq": parts[3] if len(parts) > 3 else "unknown",
                **metrics,
                "acc_matrix": acc_matrix.tolist()
            }
            
            # 添加任务准确率
            n = len(acc_matrix)
            for i in range(n):
                result[f"Task{i+1}_AfterT{i+1}"] = acc_matrix[i][i]
            for j in range(n):
                result[f"Task{j+1}_Final"] = acc_matrix[n-1][j]
            
            all_results.append(result)
            print("✓")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    if not all_results:
        print("\n没有成功提取的结果")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_csv = output_dir / "kaggle_results_summary.csv"
    df.to_csv(results_csv, index=False, encoding='utf-8')
    print(f"\n✓ 结果已保存: {results_csv}")
    
    # 显示统计
    print("\n" + "="*80)
    print("结果统计")
    print("="*80)
    print(f"\n平均 AA: {df['AA'].mean():.4f} ± {df['AA'].std():.4f}")
    print(f"平均 AIA: {df['AIA'].mean():.4f} ± {df['AIA'].std():.4f}")
    print(f"平均 FM: {df['FM'].mean():.4f} ± {df['FM'].std():.4f}")
    print(f"平均 BWT: {df['BWT'].mean():.4f} ± {df['BWT'].std():.4f}")
    print(f"平均 FWT: {df['FWT'].mean():.4f} ± {df['FWT'].std():.4f}")
    
    # 显示最佳结果
    print("\n" + "="*80)
    print("最佳结果 (按AA排序)")
    print("="*80)
    
    df_sorted = df.sort_values("AA", ascending=False)
    print("\nTop 5:")
    for idx, row in df_sorted.head(5).iterrows():
        print(f"  {row['file']}: AA={row['AA']:.4f}, FM={row['FM']:.4f}")
    
    print("\n分析完成！")


def main():
    parser = argparse.ArgumentParser(description="分析Kaggle实验结果")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Kaggle结果目录（解压后的checkpoints目录）")
    parser.add_argument("--output_dir", type=str, default="./kaggle_analysis",
                       help="分析结果输出目录")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"错误: 结果目录不存在: {results_dir}")
        return
    
    analyze_results(results_dir, output_dir)


if __name__ == "__main__":
    main()
