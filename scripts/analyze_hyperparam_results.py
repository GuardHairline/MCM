#!/usr/bin/env python3
"""
MASC 超参数搜索结果分析脚本

功能：
1. 读取所有实验的train_info.json文件
2. 提取关键指标（F1分数、准确率等）
3. 生成对比表格和可视化图表
4. 找出最佳超参数组合
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HyperparamResultAnalyzer:
    """超参数搜索结果分析器"""
    
    def __init__(self, config_index_path: str, checkpoint_dir: str = "checkpoints/hyperparam_search"):
        """
        初始化分析器
        
        Args:
            config_index_path: config_index.json文件路径
            checkpoint_dir: checkpoint目录路径（默认为checkpoints/hyperparam_search）
        """
        self.config_index_path = Path(config_index_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # 加载配置索引
        with open(self.config_index_path) as f:
            self.config_index = json.load(f)
        
        self.results = []
        
    def load_train_info(self, dataset: str, strategy: str, seq_suffix: str) -> Optional[Dict]:
        """
        加载train_info文件
        
        Args:
            dataset: 数据集名称
            strategy: 策略名称
            seq_suffix: 序列后缀（如hp1, hp2）
        
        Returns:
            train_info字典，如果文件不存在返回None
        """
        # 实际的文件命名格式: train_info_{dataset}_{strategy}_t2m_{seq_suffix}.json
        train_info_file = self.checkpoint_dir / f"train_info_{dataset}_{strategy}_t2m_{seq_suffix}.json"
        
        if not train_info_file.exists():
            print(f"警告: train_info文件不存在: {train_info_file}")
            return None
        
        try:
            with open(train_info_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"错误: 无法读取 {train_info_file}: {e}")
            return None
    
    def extract_metrics(self, train_info: Dict) -> Dict[str, float]:
        """
        从train_info中提取关键指标
        
        Args:
            train_info: train_info字典
        
        Returns:
            指标字典
        """
        metrics = {}
        
        # 提取sessions中的指标
        if "sessions" in train_info and len(train_info["sessions"]) >= 2:
            # 第一个任务（text_only）
            session0 = train_info["sessions"][0]
            if "final_metrics" in session0 and "best_metrics" in session0["final_metrics"]:
                best0 = session0["final_metrics"]["best_metrics"]
                metrics["task0_macro_f1"] = best0.get("macro_f1", best0.get("acc", 0.0))
                metrics["task0_accuracy"] = best0.get("accuracy", best0.get("micro_f1", 0.0))
            
            # 第二个任务（multimodal）
            session1 = train_info["sessions"][1]
            if "final_metrics" in session1 and "best_metrics" in session1["final_metrics"]:
                best1 = session1["final_metrics"]["best_metrics"]
                metrics["task1_macro_f1"] = best1.get("macro_f1", best1.get("acc", 0.0))
                metrics["task1_accuracy"] = best1.get("accuracy", best1.get("micro_f1", 0.0))
                
                # 提取持续学习指标（这是重点！）
                if "continual_metrics" in session1["final_metrics"]:
                    cont_metrics = session1["final_metrics"]["continual_metrics"]
                    # 主要指标
                    metrics["AA"] = cont_metrics.get("AA", 0.0)  # Average Accuracy
                    metrics["AIA"] = cont_metrics.get("AIA", 0.0)  # Average Incremental Accuracy
                    metrics["FM"] = cont_metrics.get("FM", 0.0)  # Forgetting Measure
                    metrics["BWT"] = cont_metrics.get("BWT", 0.0)  # Backward Transfer
                    metrics["FWT"] = cont_metrics.get("FWT", 0.0)  # Forward Transfer
                    # 其他可能的指标
                    metrics["ZS_ACC"] = cont_metrics.get("ZS_ACC", 0.0)  # Zero-shot Accuracy
        
        # 计算平均macro F1
        if "task0_macro_f1" in metrics and "task1_macro_f1" in metrics:
            metrics["avg_macro_f1"] = (metrics["task0_macro_f1"] + metrics["task1_macro_f1"]) / 2
        
        return metrics
    
    def analyze_all_experiments(self, dataset: str = "twitter2015") -> pd.DataFrame:
        """
        分析所有实验结果
        
        Args:
            dataset: 数据集名称
        
        Returns:
            包含所有结果的DataFrame
        """
        results = []
        
        # 计算每个策略有多少组超参数
        total_configs = len(self.config_index["configs"])
        num_strategies = len(self.config_index["strategies"])
        configs_per_strategy = total_configs // num_strategies
        
        for exp_id, config in enumerate(self.config_index["configs"], 1):
            strategy = config["strategy"]
            lr = config["lr"]
            step_size = config["step_size"]
            gamma = config["gamma"]
            
            # 计算序列后缀：每个策略从hp1开始
            hp_id = ((exp_id - 1) % configs_per_strategy) + 1
            seq_suffix = f"hp{hp_id}"
            
            # 加载train_info文件（单个文件包含两个任务）
            train_info = self.load_train_info(dataset, strategy, seq_suffix)
            
            if train_info is None:
                print(f"跳过实验 #{exp_id}: 未找到train_info文件 ({seq_suffix})")
                continue
            
            # 提取指标
            metrics = self.extract_metrics(train_info)
            
            # 构建结果行
            result = {
                "exp_id": exp_id,
                "strategy": strategy,
                "lr": lr,
                "step_size": step_size,
                "gamma": gamma,
                **metrics
            }
            
            results.append(result)
            
            # 打印进度
            avg_macro_f1 = metrics.get('avg_macro_f1', 0.0)
            aa = metrics.get('AA', 0.0)
            fm = metrics.get('FM', 0.0)
            print(f"✓ 实验 #{exp_id} ({strategy}, {seq_suffix}): "
                  f"avg_macro_F1={avg_macro_f1:.2f}, AA={aa:.2f}, FM={fm:.2f}")
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        return df
    
    def generate_summary_table(self, df: pd.DataFrame, output_file: str = "hyperparam_results.csv"):
        """
        生成汇总表格
        
        Args:
            df: 结果DataFrame
            output_file: 输出CSV文件路径
        """
        # 保存完整结果
        df.to_csv(output_file, index=False, float_format="%.4f")
        print(f"\n完整结果已保存到: {output_file}")
        
        # 生成按策略分组的汇总 - 关注持续学习指标
        agg_dict = {}
        if "avg_macro_f1" in df.columns:
            agg_dict["avg_macro_f1"] = ["mean", "std", "max", "min"]
        if "AA" in df.columns:
            agg_dict["AA"] = ["mean", "std", "max", "min"]
        if "AIA" in df.columns:
            agg_dict["AIA"] = ["mean", "std", "max", "min"]
        if "FM" in df.columns:
            agg_dict["FM"] = ["mean", "std", "max", "min"]
        
        summary_by_strategy = df.groupby("strategy").agg(agg_dict).round(4)
        
        summary_file = output_file.replace(".csv", "_by_strategy.csv")
        summary_by_strategy.to_csv(summary_file)
        print(f"按策略汇总已保存到: {summary_file}")
        
        # 打印到控制台
        print("\n" + "=" * 80)
        print("按策略汇总:")
        print("=" * 80)
        print(summary_by_strategy)
        print()
        
    def find_best_hyperparams(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        找出每个策略在不同指标下的最佳超参数
        
        Args:
            df: 结果DataFrame
        
        Returns:
            每个策略的最佳配置字典
        """
        best_configs = {}
        
        # 定义要优化的关键指标（越高越好）
        key_metrics = ["avg_macro_f1", "AA", "AIA"]
        # 定义要最小化的指标（越低越好）
        minimize_metrics = ["FM"]  # Forgetting Measure
        
        for strategy in df["strategy"].unique():
            strategy_df = df[df["strategy"] == strategy]
            
            best_configs[strategy] = {}
            
            # 对每个指标找最佳配置
            for metric in key_metrics:
                if metric in strategy_df.columns:
                    best_idx = strategy_df[metric].idxmax()
                    best_row = strategy_df.loc[best_idx]
                    
                    best_configs[strategy][f"best_by_{metric}"] = {
                        "exp_id": int(best_row["exp_id"]),
                        "lr": best_row["lr"],
                        "step_size": int(best_row["step_size"]),
                        "gamma": best_row["gamma"],
                        metric: best_row[metric],
                        "AA": best_row.get("AA", 0.0),
                        "AIA": best_row.get("AIA", 0.0),
                        "FM": best_row.get("FM", 0.0),
                        "avg_macro_f1": best_row.get("avg_macro_f1", 0.0)
                    }
            
            # 对需要最小化的指标找最佳配置
            for metric in minimize_metrics:
                if metric in strategy_df.columns:
                    best_idx = strategy_df[metric].idxmin()
                    best_row = strategy_df.loc[best_idx]
                    
                    best_configs[strategy][f"best_by_{metric}"] = {
                        "exp_id": int(best_row["exp_id"]),
                        "lr": best_row["lr"],
                        "step_size": int(best_row["step_size"]),
                        "gamma": best_row["gamma"],
                        metric: best_row[metric],
                        "AA": best_row.get("AA", 0.0),
                        "AIA": best_row.get("AIA", 0.0),
                        "FM": best_row.get("FM", 0.0),
                        "avg_macro_f1": best_row.get("avg_macro_f1", 0.0)
                    }
        
        # 打印最佳配置
        self._print_best_configs(best_configs)
        
        return best_configs
    
    def _print_best_configs(self, best_configs: Dict):
        """打印每个策略的最佳超参数配置"""
        print("\n" + "=" * 100)
        print("每个策略在不同指标下的最佳超参数配置:")
        print("=" * 100)
        
        for strategy, configs in best_configs.items():
            print(f"\n{'=' * 100}")
            print(f"{strategy.upper()} 策略:")
            print(f"{'=' * 100}")
            
            # 按avg_macro_f1最佳
            if "best_by_avg_macro_f1" in configs:
                config = configs["best_by_avg_macro_f1"]
                print(f"\n【最佳 Macro F1】")
                print(f"  实验ID: exp_{config['exp_id']}")
                print(f"  超参数: lr={config['lr']:.0e}, step_size={config['step_size']}, gamma={config['gamma']}")
                print(f"  Macro F1: {config['avg_macro_f1']:.4f}")
                print(f"  AA: {config['AA']:.4f}, AIA: {config['AIA']:.4f}, FM: {config['FM']:.4f}")
            
            # 按AA最佳
            if "best_by_AA" in configs:
                config = configs["best_by_AA"]
                print(f"\n【最佳 AA (Average Accuracy)】")
                print(f"  实验ID: exp_{config['exp_id']}")
                print(f"  超参数: lr={config['lr']:.0e}, step_size={config['step_size']}, gamma={config['gamma']}")
                print(f"  AA: {config['AA']:.4f}")
                print(f"  Macro F1: {config['avg_macro_f1']:.4f}, AIA: {config['AIA']:.4f}, FM: {config['FM']:.4f}")
            
            # 按AIA最佳
            if "best_by_AIA" in configs:
                config = configs["best_by_AIA"]
                print(f"\n【最佳 AIA (Average Incremental Accuracy)】")
                print(f"  实验ID: exp_{config['exp_id']}")
                print(f"  超参数: lr={config['lr']:.0e}, step_size={config['step_size']}, gamma={config['gamma']}")
                print(f"  AIA: {config['AIA']:.4f}")
                print(f"  Macro F1: {config['avg_macro_f1']:.4f}, AA: {config['AA']:.4f}, FM: {config['FM']:.4f}")
            
            # 按FM最小（最少遗忘）
            if "best_by_FM" in configs:
                config = configs["best_by_FM"]
                print(f"\n【最小 FM (Forgetting Measure) - 最少遗忘】")
                print(f"  实验ID: exp_{config['exp_id']}")
                print(f"  超参数: lr={config['lr']:.0e}, step_size={config['step_size']}, gamma={config['gamma']}")
                print(f"  FM: {config['FM']:.4f} (越低越好)")
                print(f"  Macro F1: {config['avg_macro_f1']:.4f}, AA: {config['AA']:.4f}, AIA: {config['AIA']:.4f}")
        
        print(f"\n{'=' * 100}")
    
    def plot_results(self, df: pd.DataFrame, output_dir: str = "hyperparam_analysis"):
        """
        生成可视化图表
        
        Args:
            df: 结果DataFrame
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图风格
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # 1. 不同策略的F1分数箱线图
        self._plot_f1_by_strategy(df, output_path)
        
        # 2. 每个策略的超参数热力图
        self._plot_heatmap_by_strategy(df, output_path)
        
        # 3. 学习率对性能的影响
        self._plot_lr_impact(df, output_path)
        
        # 4. Step Size和Gamma的影响
        self._plot_step_gamma_impact(df, output_path)
        
        # 5. Task0和Task1的F1对比
        self._plot_task_comparison(df, output_path)
        
        print(f"\n所有图表已保存到: {output_path}")
    
    def _plot_f1_by_strategy(self, df: pd.DataFrame, output_path: Path):
        """绘制不同策略的持续学习指标对比图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Macro F1
        if "avg_macro_f1" in df.columns:
            sns.boxplot(data=df, x="strategy", y="avg_macro_f1", ax=ax1)
            ax1.set_title("Average Macro F1 by Strategy", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Strategy", fontsize=12)
            ax1.set_ylabel("Average Macro F1", fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
        
        # AA (Average Accuracy)
        if "AA" in df.columns:
            sns.boxplot(data=df, x="strategy", y="AA", ax=ax2)
            ax2.set_title("AA (Average Accuracy) by Strategy", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Strategy", fontsize=12)
            ax2.set_ylabel("AA", fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
        
        # AIA (Average Incremental Accuracy)
        if "AIA" in df.columns:
            sns.boxplot(data=df, x="strategy", y="AIA", ax=ax3)
            ax3.set_title("AIA (Average Incremental Accuracy) by Strategy", fontsize=14, fontweight='bold')
            ax3.set_xlabel("Strategy", fontsize=12)
            ax3.set_ylabel("AIA", fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
        
        # FM (Forgetting Measure - 越低越好)
        if "FM" in df.columns:
            sns.boxplot(data=df, x="strategy", y="FM", ax=ax4)
            ax4.set_title("FM (Forgetting Measure) by Strategy (Lower is Better)", 
                         fontsize=14, fontweight='bold')
            ax4.set_xlabel("Strategy", fontsize=12)
            ax4.set_ylabel("FM (Lower is Better)", fontsize=12)
            ax4.tick_params(axis='x', rotation=45)
            # 添加参考线在0处
            ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path / "continual_metrics_by_strategy.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 生成图表: continual_metrics_by_strategy.png")
    
    def _plot_heatmap_by_strategy(self, df: pd.DataFrame, output_path: Path):
        """为每个策略绘制超参数热力图 - 显示avg_macro_f1"""
        strategies = df["strategy"].unique()
        n_strategies = len(strategies)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        # 选择用于热力图的指标
        metric_to_plot = "avg_macro_f1" if "avg_macro_f1" in df.columns else "AA"
        
        for idx, strategy in enumerate(strategies):
            strategy_df = df[df["strategy"] == strategy]
            
            # 创建pivot table
            pivot = strategy_df.pivot_table(
                values=metric_to_plot,
                index="lr",
                columns="step_size",
                aggfunc="mean"
            )
            
            # 绘制热力图
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                ax=axes[idx],
                cbar_kws={"label": metric_to_plot.replace("_", " ").title()}
            )
            axes[idx].set_title(f"{strategy.upper()} - {metric_to_plot.upper()} Heatmap", 
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel("Step Size", fontsize=10)
            axes[idx].set_ylabel("Learning Rate", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / "heatmap_macro_f1_by_strategy.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 生成图表: heatmap_macro_f1_by_strategy.png")
        
        # 为FM（遗忘）生成另一个热力图
        if "FM" in df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            
            for idx, strategy in enumerate(strategies):
                strategy_df = df[df["strategy"] == strategy]
                
                pivot = strategy_df.pivot_table(
                    values="FM",
                    index="lr",
                    columns="step_size",
                    aggfunc="mean"
                )
                
                # 使用反向色彩映射，因为FM越低越好
                sns.heatmap(
                    pivot,
                    annot=True,
                    fmt=".2f",
                    cmap="RdYlGn_r",  # 红色=高遗忘（差），绿色=低遗忘（好）
                    ax=axes[idx],
                    cbar_kws={"label": "FM (Lower is Better)"}
                )
                axes[idx].set_title(f"{strategy.upper()} - FM (Forgetting) Heatmap", 
                                   fontsize=12, fontweight='bold')
                axes[idx].set_xlabel("Step Size", fontsize=10)
                axes[idx].set_ylabel("Learning Rate", fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_path / "heatmap_forgetting_by_strategy.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 生成图表: heatmap_forgetting_by_strategy.png")
    
    def _plot_lr_impact(self, df: pd.DataFrame, output_path: Path):
        """绘制学习率对性能的影响 - 使用持续学习指标"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, strategy in enumerate(df["strategy"].unique()):
            strategy_df = df[df["strategy"] == strategy]
            
            # 按学习率分组 - 使用新指标
            agg_dict = {}
            if "avg_macro_f1" in strategy_df.columns:
                agg_dict["avg_macro_f1"] = ["mean", "std"]
            if "AA" in strategy_df.columns:
                agg_dict["AA"] = ["mean", "std"]
            if "FM" in strategy_df.columns:
                agg_dict["FM"] = ["mean", "std"]
            
            lr_grouped = strategy_df.groupby("lr").agg(agg_dict)
            
            x = np.arange(len(lr_grouped))
            lr_labels = [f"{lr:.0e}" for lr in lr_grouped.index]
            
            # 绘制指标
            ax = axes[idx]
            
            if "avg_macro_f1" in strategy_df.columns:
                ax.errorbar(x, lr_grouped["avg_macro_f1"]["mean"], 
                           yerr=lr_grouped["avg_macro_f1"]["std"],
                           marker='o', label='Macro F1', linewidth=2, markersize=8)
            
            if "AA" in strategy_df.columns:
                ax.errorbar(x, lr_grouped["AA"]["mean"], 
                           yerr=lr_grouped["AA"]["std"],
                           marker='s', label='AA', linewidth=2, markersize=8)
            
            ax.set_title(f"{strategy.upper()} - Learning Rate Impact", 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel("Learning Rate", fontsize=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(lr_labels)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "lr_impact.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 生成图表: lr_impact.png")
    
    def _plot_step_gamma_impact(self, df: pd.DataFrame, output_path: Path):
        """绘制Step Size和Gamma的影响 - 使用Macro F1"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metric_to_plot = "avg_macro_f1" if "avg_macro_f1" in df.columns else "AA"
        
        for idx, strategy in enumerate(df["strategy"].unique()):
            strategy_df = df[df["strategy"] == strategy]
            
            # 创建scatter plot
            ax = axes[idx]
            
            # 按gamma着色
            for gamma in sorted(strategy_df["gamma"].unique()):
                gamma_df = strategy_df[strategy_df["gamma"] == gamma]
                ax.scatter(gamma_df["step_size"], gamma_df[metric_to_plot],
                          label=f"γ={gamma}", s=100, alpha=0.7)
            
            ax.set_title(f"{strategy.upper()} - Step Size vs {metric_to_plot.upper()} (colored by Gamma)", 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel("Step Size", fontsize=10)
            ax.set_ylabel(metric_to_plot.replace("_", " ").title(), fontsize=10)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "step_gamma_impact.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 生成图表: step_gamma_impact.png")
    
    def _plot_task_comparison(self, df: pd.DataFrame, output_path: Path):
        """绘制Task0和Task1的Macro F1对比"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        if "task0_macro_f1" not in df.columns or "task1_macro_f1" not in df.columns:
            print("  ⚠ 跳过task_comparison.png: 缺少task0_macro_f1或task1_macro_f1列")
            return
        
        for idx, strategy in enumerate(df["strategy"].unique()):
            strategy_df = df[df["strategy"] == strategy]
            
            ax = axes[idx]
            
            # 散点图：Task0 Macro F1 vs Task1 Macro F1，用FM着色
            color_metric = "FM" if "FM" in strategy_df.columns else "avg_macro_f1"
            
            scatter = ax.scatter(strategy_df["task0_macro_f1"], strategy_df["task1_macro_f1"],
                               c=strategy_df[color_metric], 
                               cmap="RdYlGn_r" if color_metric == "FM" else "viridis",
                               s=100, alpha=0.7)
            
            # 添加对角线（理想情况）
            min_val = min(strategy_df["task0_macro_f1"].min(), strategy_df["task1_macro_f1"].min())
            max_val = max(strategy_df["task0_macro_f1"].max(), strategy_df["task1_macro_f1"].max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', alpha=0.5, label='Perfect Balance')
            
            ax.set_title(f"{strategy.upper()} - Task Performance Comparison", 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel("Task 0 Macro F1 (Text Only)", fontsize=10)
            ax.set_ylabel("Task 1 Macro F1 (Multimodal)", fontsize=10)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # 添加colorbar
            plt.colorbar(scatter, ax=ax, label=color_metric)
        
        plt.tight_layout()
        plt.savefig(output_path / "task_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 生成图表: task_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="分析MASC超参数搜索结果")
    parser.add_argument("--config_index", type=str,
                       default="scripts/configs/hyperparam_search/config_index.json",
                       help="config_index.json文件路径")
    parser.add_argument("--checkpoint_dir", type=str,
                       default="checkpoints/hyperparam_search",
                       help="checkpoint目录路径（默认：checkpoints/hyperparam_search）")
    parser.add_argument("--dataset", type=str,
                       default="twitter2015",
                       help="数据集名称")
    parser.add_argument("--output_dir", type=str,
                       default="hyperparam_analysis",
                       help="输出目录")
    parser.add_argument("--no_plot", action="store_true",
                       help="不生成图表")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MASC 超参数搜索结果分析")
    print("=" * 80)
    print()
    
    # 创建分析器
    analyzer = HyperparamResultAnalyzer(
        config_index_path=args.config_index,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # 分析所有实验
    print("正在分析所有实验结果...")
    df = analyzer.analyze_all_experiments(dataset=args.dataset)
    
    if df.empty:
        print("\n错误: 未找到任何实验结果")
        return
    
    print(f"\n成功加载 {len(df)} 个实验结果")
    
    # 生成汇总表格
    output_csv = f"{args.output_dir}/hyperparam_results.csv"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    analyzer.generate_summary_table(df, output_csv)
    
    # 找出最佳超参数
    best_configs = analyzer.find_best_hyperparams(df)
    
    # 保存最佳配置到JSON
    best_configs_file = f"{args.output_dir}/best_hyperparams.json"
    with open(best_configs_file, 'w') as f:
        json.dump(best_configs, f, indent=2)
    print(f"\n最佳配置已保存到: {best_configs_file}")
    
    # 生成可视化图表
    if not args.no_plot:
        print("\n正在生成可视化图表...")
        analyzer.plot_results(df, args.output_dir)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n结果文件:")
    print(f"  - CSV表格: {output_csv}")
    print(f"  - 最佳配置: {best_configs_file}")
    if not args.no_plot:
        print(f"  - 可视化图表: {args.output_dir}/*.png")
    print()


if __name__ == "__main__":
    main()

