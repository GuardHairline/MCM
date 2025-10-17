# utils/experiment_utils.py
"""
实验工具函数

提供：
1. 实验配置管理
2. 结果记录和可视化
3. 消融实验支持
4. 对比实验支持
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 基础配置
    experiment_name: str
    task_order: List[str]
    fusion_strategies: List[str] = None  # 用于对比实验
    cl_methods: List[str] = None  # 用于对比实验
    
    # 消融实验配置
    ablation_components: List[str] = None  # 要消融的组件
    
    # 其他配置
    num_runs: int = 1  # 重复运行次数
    seed: int = 42
    output_dir: str = "experiments"
    
    def __post_init__(self):
        if self.fusion_strategies is None:
            self.fusion_strategies = ["gated"]
        if self.cl_methods is None:
            self.cl_methods = ["ewc"]
        if self.ablation_components is None:
            self.ablation_components = []


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, experiment_name: str, output_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.exp_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.results = []
        self.start_time = time.time()
        
        logger.info(f"ExperimentLogger initialized: {self.exp_dir}")
    
    def log_config(self, config: Dict[str, Any]):
        """记录实验配置"""
        config_file = os.path.join(self.exp_dir, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Config saved to {config_file}")
    
    def log_result(self, run_id: int, task_name: str, metrics: Dict[str, float]):
        """记录单次运行结果"""
        result = {
            'run_id': run_id,
            'task_name': task_name,
            'metrics': metrics,
            'timestamp': time.time()
        }
        self.results.append(result)
        
        # 实时保存
        self._save_results()
    
    def _save_results(self):
        """保存结果"""
        results_file = os.path.join(self.exp_dir, "results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def summarize(self) -> Dict[str, Any]:
        """生成实验总结"""
        if not self.results:
            return {}
        
        # 按任务聚合结果
        task_results = {}
        for result in self.results:
            task = result['task_name']
            if task not in task_results:
                task_results[task] = []
            task_results[task].append(result['metrics'])
        
        # 计算平均值和标准差
        summary = {}
        for task, metrics_list in task_results.items():
            summary[task] = {}
            if metrics_list:
                # 计算每个指标的平均值
                metric_keys = metrics_list[0].keys()
                for key in metric_keys:
                    values = [m[key] for m in metrics_list if key in m]
                    if values:
                        summary[task][f"{key}_mean"] = sum(values) / len(values)
                        if len(values) > 1:
                            mean = sum(values) / len(values)
                            var = sum((x - mean) ** 2 for x in values) / len(values)
                            summary[task][f"{key}_std"] = var ** 0.5
        
        # 保存总结
        summary_file = os.path.join(self.exp_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - self.start_time
        logger.info(f"Experiment completed in {elapsed:.2f}s")
        logger.info(f"Summary saved to {summary_file}")
        
        return summary


class AblationStudy:
    """消融实验管理器"""
    
    def __init__(self, base_config: Dict[str, Any], components: List[str]):
        """
        Args:
            base_config: 基础配置
            components: 要消融的组件列表
        """
        self.base_config = base_config.copy()
        self.components = components
        self.configs = []
        
        # 生成消融配置
        self._generate_configs()
    
    def _generate_configs(self):
        """生成消融配置"""
        # 完整配置
        self.configs.append({
            'name': 'full',
            'config': self.base_config.copy(),
            'description': 'Full model with all components'
        })
        
        # 每次移除一个组件
        for component in self.components:
            config = self.base_config.copy()
            config[component] = False  # 禁用该组件
            self.configs.append({
                'name': f'no_{component}',
                'config': config,
                'description': f'Without {component}'
            })
        
        logger.info(f"Generated {len(self.configs)} ablation configurations")
    
    def get_configs(self) -> List[Dict[str, Any]]:
        """获取所有配置"""
        return self.configs


def create_comparison_experiments(
    base_config: Dict[str, Any],
    comparison_key: str,
    comparison_values: List[Any]
) -> List[Dict[str, Any]]:
    """
    创建对比实验配置
    
    Args:
        base_config: 基础配置
        comparison_key: 要对比的配置项（如'fusion_strategy'）
        comparison_values: 要对比的值列表
    
    Returns:
        configs: 实验配置列表
    """
    configs = []
    for value in comparison_values:
        config = base_config.copy()
        config[comparison_key] = value
        configs.append({
            'name': f'{comparison_key}_{value}',
            'config': config,
            'description': f'{comparison_key} = {value}'
        })
    
    logger.info(f"Generated {len(configs)} comparison configurations for {comparison_key}")
    return configs


def load_experiment_results(experiment_dir: str) -> Dict[str, Any]:
    """
    加载实验结果
    
    Args:
        experiment_dir: 实验目录
    
    Returns:
        results: 实验结果字典
    """
    results = {}
    
    # 加载配置
    config_file = os.path.join(experiment_dir, "config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            results['config'] = json.load(f)
    
    # 加载结果
    results_file = os.path.join(experiment_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            results['results'] = json.load(f)
    
    # 加载总结
    summary_file = os.path.join(experiment_dir, "summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            results['summary'] = json.load(f)
    
    return results

