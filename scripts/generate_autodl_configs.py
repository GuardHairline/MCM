#!/usr/bin/env python3
"""
生成AutoDL服务器配置文件

AutoDL特点：
1. 付费云GPU服务器，按时计费
2. 独占GPU，不需要等待
3. 需要在完成后关机节省费用

使用方法:
    python scripts/generate_autodl_configs.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.generate_task_config import TaskConfigGenerator


class AutoDLConfigGenerator(TaskConfigGenerator):
    """AutoDL配置生成器"""
    
    def __init__(self):
        super().__init__()
        
        # 获取当前日期作为文件夹名
        from datetime import datetime
        date_folder = datetime.now().strftime("%y%m%d")
        
        # AutoDL环境配置 - 使用数据盘 /root/autodl-tmp/
        autodl_base = f"/root/autodl-tmp/checkpoints/{date_folder}"
        
        self.environments["autodl"] = {
            "base_dir": "",
            "model_name": f"{autodl_base}/{{task}}_{{dataset}}_{{strategy}}_{{seq}}.pt",
            "log_dir": f"{autodl_base}/log",
            "checkpoint_dir": autodl_base,
            "ewc_dir": f"{autodl_base}/ewc_params",
            "gem_dir": f"{autodl_base}/gem_memory"
        }
        
        # AutoDL任务配置（更大的batch size）
        self.autodl_task_configs = {
            "batch_size": 16,  # AutoDL GPU
            "num_workers": 4,   # 适中的worker数
        }
    
    def generate_autodl_configs(self, output_dir: str = "scripts/configs/autodl_config"):
        """
        生成AutoDL配置文件
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取日期文件夹
        from datetime import datetime
        date_folder = datetime.now().strftime("%y%m%d")
        
        print("="*80)
        print("生成AutoDL配置文件")
        print("="*80)
        print()
        print(f"存储位置: /root/autodl-tmp/checkpoints/{date_folder}/")
        print()
        
        # 任务顺序：seq1 -> seq2, 对于每个seq：twitter2015 -> twitter2017 -> mix
        # 策略顺序：deqa -> none -> moe -> replay -> ewc -> lwf -> si -> mas -> gem
        sequences = ["seq1", "seq2"]
        datasets = ["twitter2015", "twitter2017", "mix"]
        strategies = ["deqa", "none", "moe", "replay", "ewc", "lwf", "si", "mas", "gem"]
        
        config_list = []
        
        for seq in sequences:
            for dataset in datasets:
                for strategy in strategies:
                    config_name = f"autodl_{dataset}_{strategy}_{seq}.json"
                    config_path = output_path / config_name
                    
                    # 创建配置
                    config = self.create_task_config(
                        task_name="all",  # 将生成所有任务
                        session_name=f"{dataset}_{strategy}_{seq}",
                        dataset=dataset,
                        env="autodl",
                        strategy=strategy,
                        mode="t2m" if seq == "seq1" else "m2t"
                    )
                    
                    # AutoDL优化：更大的batch size
                    if config and "tasks" in config:
                        for task in config["tasks"]:
                            task["batch_size"] = self.autodl_task_configs["batch_size"]
                        
                        if "global_params" in config:
                            config["global_params"]["num_workers"] = self.autodl_task_configs["num_workers"]
                    
                    # 保存配置
                    if config:
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config, f, indent=2, ensure_ascii=False)
                        
                        config_list.append({
                            "name": config_name,
                            "path": str(config_path),
                            "dataset": dataset,
                            "strategy": strategy,
                            "sequence": seq
                        })
                        
                        print(f"✓ {config_name}")
        
        # 生成索引文件
        index_path = output_path / "config_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total": len(config_list),
                "configs": config_list,
                "execution_order": "seq1->seq2, for each seq: twitter2015->twitter2017->mix, for each dataset: deqa->none->moe->replay->ewc->lwf->si->mas->gem"
            }, f, indent=2)
        
        print()
        print(f"总计生成 {len(config_list)} 个配置文件")
        print(f"索引文件: {index_path}")
        print()
        
        return config_list


def main():
    generator = AutoDLConfigGenerator()
    configs = generator.generate_autodl_configs()
    
    print("="*80)
    print("配置生成完成！")
    print("="*80)
    print()
    print("下一步：")
    print("1. 上传代码到AutoDL服务器")
    print("2. 运行: bash scripts/configs/autodl_config/run_autodl_experiments.sh")
    print()
    print("⚠️  重要提示：")
    print("- AutoDL是付费服务，任务完成后会自动关机")
    print("- 请确保数据文件已上传")
    print("- 建议先运行单个配置测试")
    print()


if __name__ == "__main__":
    main()

