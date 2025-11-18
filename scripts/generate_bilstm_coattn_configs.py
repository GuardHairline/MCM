#!/usr/bin/env python3
"""
生成BiLSTM+Co-Attention对比实验配置文件

用于对比以下方案：
1. Baseline: 简单拼接 + Linear
2. +BiLSTM: 简单拼接 + BiLSTM + CRF
3. +Gate: 门控融合 + BiLSTM + CRF
4. +CoAttn: Co-Attention融合 + BiLSTM + CRF
5. +Adaptive: 自适应融合 + BiLSTM + CRF

使用方法:
    python scripts/generate_bilstm_coattn_configs.py \
        --dataset twitter2015 \
        --task mner \
        --output_dir scripts/configs/bilstm_coattn_comparison
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


class BiLSTMCoAttnConfigGenerator:
    """BiLSTM + Co-Attention对比实验配置生成器"""
    
    def __init__(self):
        # 基础配置
        self.base_config = {
            "model_type": "base_multimodal",
            "text_encoder": "microsoft/deberta-v3-base",
            "img_encoder": "google/vit-base-patch16-224",
            
            "learning_rate": 2e-5,
            "num_epochs": 20,
            "batch_size": 16,
            "max_seq_length": 128,
            "warmup_ratio": 0.1,
            
            "dropout": 0.3,
            "weight_decay": 0.01,
            
            "scheduler_type": "cosine",
            "save_steps": 500,
            "eval_steps": 100,
            "logging_steps": 50,
            
            "seed": 42,
            "use_amp": True
        }
        
        # 实验方案
        self.experiment_variants = {
            "baseline": {
                "name": "Baseline",
                "description": "简单拼接 + Linear（无BiLSTM，无CRF）",
                "config": {
                    "fusion_type": "concat",
                    "use_bilstm_head": 0,
                    "use_crf": 0
                },
                "expected_f1": 35
            },
            "bilstm": {
                "name": "+BiLSTM",
                "description": "简单拼接 + BiLSTM + CRF",
                "config": {
                    "fusion_type": "concat",
                    "use_bilstm_head": 1,
                    "bilstm_hidden_size": 256,
                    "bilstm_num_layers": 2,
                    "use_crf": 1
                },
                "expected_f1": 57
            },
            "gate": {
                "name": "+Gate",
                "description": "门控融合 + BiLSTM + CRF",
                "config": {
                    "fusion_type": "gate",
                    "attn_heads": 8,
                    "use_bilstm_head": 1,
                    "bilstm_hidden_size": 256,
                    "bilstm_num_layers": 2,
                    "use_crf": 1
                },
                "expected_f1": 64
            },
            "coattn": {
                "name": "+CoAttn",
                "description": "Co-Attention融合 + BiLSTM + CRF",
                "config": {
                    "fusion_type": "coattn",
                    "attn_heads": 8,
                    "use_bilstm_head": 1,
                    "bilstm_hidden_size": 256,
                    "bilstm_num_layers": 2,
                    "use_crf": 1
                },
                "expected_f1": 69
            },
            "adaptive": {
                "name": "+Adaptive",
                "description": "自适应融合 + BiLSTM + CRF",
                "config": {
                    "fusion_type": "adaptive",
                    "attn_heads": 8,
                    "use_bilstm_head": 1,
                    "bilstm_hidden_size": 256,
                    "bilstm_num_layers": 2,
                    "use_crf": 1
                },
                "expected_f1": 70
            }
        }
        
        # 任务特定配置
        self.task_configs = {
            "mner": {
                "task_name": "mner",
                "num_labels": 9,
                "label_list": ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", 
                              "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
            },
            "mate": {
                "task_name": "mate",
                "num_labels": 3,
                "label_list": ["O", "B-ASP", "I-ASP"]
            },
            "mabsa": {
                "task_name": "mabsa",
                "num_labels": 7,
                "label_list": ["O", "B-POS", "I-POS", "B-NEG", "I-NEG", 
                              "B-NEU", "I-NEU"]
            }
        }
        
        # 数据集路径
        self.dataset_paths = {
            "twitter2015": {
                "train_text": "data/MNER/twitter2015/train.txt",
                "dev_text": "data/MNER/twitter2015/valid.txt",
                "test_text": "data/MNER/twitter2015/test.txt",
                "image_dir": "data/twitter2015_images"
            },
            "twitter2017": {
                "train_text": "data/MNER/twitter2017/train.txt",
                "dev_text": "data/MNER/twitter2017/valid.txt",
                "test_text": "data/MNER/twitter2017/test.txt",
                "image_dir": "data/twitter2017_images"
            }
        }
    
    def generate_config(
        self,
        variant: str,
        dataset: str,
        task: str,
        mode: str = "multimodal"
    ) -> Dict[str, Any]:
        """
        生成单个实验配置
        
        参数:
            variant: 实验方案 ('baseline', 'bilstm', 'gate', 'coattn', 'adaptive')
            dataset: 数据集名称 ('twitter2015', 'twitter2017')
            task: 任务名称 ('mner', 'mate', 'mabsa')
            mode: 模式 ('text_only', 'multimodal')
        
        返回:
            配置字典
        """
        if variant not in self.experiment_variants:
            raise ValueError(f"Unknown variant: {variant}")
        if dataset not in self.dataset_paths:
            raise ValueError(f"Unknown dataset: {dataset}")
        if task not in self.task_configs:
            raise ValueError(f"Unknown task: {task}")
        
        # 合并配置
        config = self.base_config.copy()
        config.update(self.task_configs[task])
        config.update(self.dataset_paths[dataset])
        config.update(self.experiment_variants[variant]["config"])
        
        # 添加元信息
        config["experiment_name"] = f"{dataset}_{task}_{variant}_{mode}"
        config["variant"] = variant
        config["variant_name"] = self.experiment_variants[variant]["name"]
        config["variant_description"] = self.experiment_variants[variant]["description"]
        config["expected_f1"] = self.experiment_variants[variant]["expected_f1"]
        config["dataset"] = dataset
        config["mode"] = mode
        
        # 输出目录
        config["output_dir"] = f"checkpoints/bilstm_coattn/{dataset}_{task}_{variant}_{mode}"
        
        return config
    
    def generate_all_configs(
        self,
        dataset: str,
        task: str,
        variants: List[str] = None,
        modes: List[str] = None,
        output_dir: str = "scripts/configs/bilstm_coattn_comparison"
    ):
        """
        生成所有实验配置
        
        参数:
            dataset: 数据集名称
            task: 任务名称
            variants: 要生成的实验方案列表（默认全部）
            modes: 要生成的模式列表（默认全部）
            output_dir: 输出目录
        """
        if variants is None:
            variants = list(self.experiment_variants.keys())
        if modes is None:
            modes = ["text_only", "multimodal"]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        print(f"\n{'='*80}")
        print(f"生成BiLSTM+Co-Attention对比实验配置")
        print(f"{'='*80}")
        print(f"数据集: {dataset}")
        print(f"任务: {task}")
        print(f"实验方案: {', '.join(variants)}")
        print(f"模式: {', '.join(modes)}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*80}\n")
        
        for variant in variants:
            for mode in modes:
                # 生成配置
                config = self.generate_config(variant, dataset, task, mode)
                
                # 保存文件
                filename = f"{dataset}_{task}_{variant}_{mode}.json"
                filepath = output_path / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                generated_files.append(filepath)
                
                # 打印信息
                variant_info = self.experiment_variants[variant]
                print(f"✓ {variant_info['name']:12s} ({mode:12s}): {filename}")
                print(f"  {variant_info['description']}")
                print(f"  预期F1: {variant_info['expected_f1']}%")
                print()
        
        # 生成索引文件
        index_file = output_path / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({
                "description": "BiLSTM + Co-Attention对比实验",
                "dataset": dataset,
                "task": task,
                "variants": variants,
                "modes": modes,
                "total_configs": len(generated_files),
                "configs": [str(f.relative_to(output_path)) for f in generated_files]
            }, f, indent=2, ensure_ascii=False)
        
        print(f"{'='*80}")
        print(f"✅ 配置生成完成")
        print(f"{'='*80}")
        print(f"总配置数: {len(generated_files)}")
        print(f"索引文件: {index_file}")
        print(f"{'='*80}\n")
        
        # 生成运行脚本
        self._generate_runner_script(output_path, generated_files)
        
        # 生成README
        self._generate_readme(output_path, dataset, task, variants, modes)
        
        return generated_files
    
    def _generate_runner_script(self, output_path: Path, config_files: List[Path]):
        """生成批量运行脚本"""
        
        script_path = output_path / "run_all.sh"
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# BiLSTM + Co-Attention对比实验批量运行脚本\n\n")
            f.write("echo \"开始运行对比实验...\"\n")
            f.write("echo \"总配置数: {}\"\n\n".format(len(config_files)))
            
            for i, config_file in enumerate(config_files, 1):
                rel_path = config_file.relative_to(Path.cwd())
                f.write(f"echo \"[{i}/{len(config_files)}] 运行: {config_file.name}\"\n")
                f.write(f"python -m scripts.train_with_zero_shot --config {rel_path}\n\n")
            
            f.write("echo \"所有实验完成！\"\n")
        
        # 设置可执行权限
        script_path.chmod(0o755)
        
        print(f"✓ 运行脚本: {script_path}")
    
    def _generate_readme(
        self,
        output_path: Path,
        dataset: str,
        task: str,
        variants: List[str],
        modes: List[str]
    ):
        """生成README"""
        
        readme_path = output_path / "README.md"
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# BiLSTM + Co-Attention对比实验\n\n")
            f.write(f"## 实验信息\n\n")
            f.write(f"- **数据集**: {dataset}\n")
            f.write(f"- **任务**: {task}\n")
            f.write(f"- **实验方案**: {len(variants)}种\n")
            f.write(f"- **模式**: {', '.join(modes)}\n")
            f.write(f"- **总配置数**: {len(variants) * len(modes)}\n\n")
            
            f.write(f"## 实验方案\n\n")
            f.write("| 方案 | 描述 | 预期F1 |\n")
            f.write("|------|------|--------|\n")
            for variant in variants:
                info = self.experiment_variants[variant]
                f.write(f"| **{info['name']}** | {info['description']} | {info['expected_f1']}% |\n")
            
            f.write(f"\n## 使用方法\n\n")
            f.write(f"### 运行单个实验\n\n")
            f.write(f"```bash\n")
            f.write(f"python -m scripts.train_with_zero_shot \\\n")
            f.write(f"    --config {output_path.name}/{dataset}_{task}_coattn_multimodal.json\n")
            f.write(f"```\n\n")
            
            f.write(f"### 批量运行所有实验\n\n")
            f.write(f"```bash\n")
            f.write(f"bash {output_path.name}/run_all.sh\n")
            f.write(f"```\n\n")
            
            f.write(f"## 预期结果\n\n")
            f.write(f"### {dataset.upper()} {task.upper()}\n\n")
            f.write("| 方法 | Token Acc | Chunk F1 | 相对提升 |\n")
            f.write("|------|-----------|----------|----------|\n")
            baseline_f1 = self.experiment_variants["baseline"]["expected_f1"]
            for variant in variants:
                info = self.experiment_variants[variant]
                improvement = info["expected_f1"] - baseline_f1
                f.write(f"| {info['name']} | - | {info['expected_f1']}% | +{improvement}% |\n")
            
            f.write(f"\n## 文件列表\n\n")
            f.write(f"```\n")
            for variant in variants:
                for mode in modes:
                    filename = f"{dataset}_{task}_{variant}_{mode}.json"
                    f.write(f"- {filename}\n")
            f.write(f"```\n\n")
            
            f.write(f"## 注意事项\n\n")
            f.write(f"1. 确保已安装所有依赖：`pip install -r requirements.txt`\n")
            f.write(f"2. 确保数据集路径正确\n")
            f.write(f"3. 实验结果保存在 `checkpoints/bilstm_coattn/` 目录\n")
            f.write(f"4. 每个实验预计需要1-2小时\n")
        
        print(f"✓ README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="生成BiLSTM+Co-Attention对比实验配置"
    )
    parser.add_argument("--dataset", type=str, default="twitter2015",
                       choices=["twitter2015", "twitter2017"],
                       help="数据集名称")
    parser.add_argument("--task", type=str, default="mner",
                       choices=["mner", "mate", "mabsa"],
                       help="任务名称")
    parser.add_argument("--variants", type=str, nargs="+",
                       choices=["baseline", "bilstm", "gate", "coattn", "adaptive"],
                       help="实验方案（默认全部）")
    parser.add_argument("--modes", type=str, nargs="+",
                       choices=["text_only", "multimodal"],
                       help="模式（默认全部）")
    parser.add_argument("--output_dir", type=str,
                       default="scripts/configs/bilstm_coattn_comparison",
                       help="输出目录")
    
    args = parser.parse_args()
    
    generator = BiLSTMCoAttnConfigGenerator()
    generator.generate_all_configs(
        dataset=args.dataset,
        task=args.task,
        variants=args.variants,
        modes=args.modes,
        output_dir=args.output_dir
    )
    
    print("\n✅ 完成！现在你可以运行实验：")
    print(f"   bash {args.output_dir}/run_all.sh\n")


if __name__ == "__main__":
    main()









