#!/usr/bin/env python3
"""
生成CRF修复测试配置 - 用于验证新代码修改的效果

专门测试CRF、valid_len修复和span loss在三个序列任务上的表现：
- MATE (Multimodal Aspect Term Extraction)
- MNER (Multimodal Named Entity Recognition)
- MABSA (Multimodal Aspect-Based Sentiment Analysis)

只包含3个实验（每个任务一个），用于快速验证修复效果。
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_task_config import TaskConfigGenerator


class CRFTestConfigGenerator:
    """CRF修复测试配置生成器"""
    
    def __init__(self):
        self.base_generator = TaskConfigGenerator()
        
        # 测试任务列表
        self.tasks = ["mate", "mner", "mabsa"]
        
        # 使用推荐的超参数（基于之前的经验）
        self.recommended_hyperparams = {
            "lr": 1e-5,           # 学习率
            "step_size": 10,      # 学习率衰减步长
            "gamma": 0.5,         # 学习率衰减因子
            "epochs": 20,         # 训练轮数
            "patience": 5,        # early stopping patience
            "batch_size": 16      # batch size
        }
        
        # 4种对比配置
        self.ablation_configs = {
            "baseline": {
                "use_crf": 0,                 # 都不启用
                "use_span_loss": 0,
                "boundary_weight": 0.2,
                "span_f1_weight": 0.0,
                "transition_weight": 0.0,
                "description": "Baseline (no CRF, no Span Loss)"
            },
            "crf_only": {
                "use_crf": 1,                 # 只启用CRF
                "use_span_loss": 0,
                "boundary_weight": 0.2,
                "span_f1_weight": 0.0,
                "transition_weight": 0.0,
                "description": "CRF only"
            },
            "span_only": {
                "use_crf": 0,                 # 只启用Span Loss
                "use_span_loss": 1,
                "boundary_weight": 0.2,
                "span_f1_weight": 0.0,
                "transition_weight": 0.0,
                "description": "Span Loss only"
            },
            "crf_and_span": {
                "use_crf": 1,                 # 都启用（推荐）
                "use_span_loss": 1,
                "boundary_weight": 0.2,
                "span_f1_weight": 0.0,
                "transition_weight": 0.0,
                "description": "CRF + Span Loss (recommended)"
            }
        }
    
    def generate_single_task_config(self,
                                    env: str,
                                    dataset: str,
                                    task_name: str,
                                    mode: str = "multimodal",
                                    use_label_embedding: bool = False,
                                    output_suffix: str = "",
                                    kaggle_mode: bool = False,
                                    ablation_type: str = "crf_and_span") -> Dict[str, Any]:
        """
        生成单个任务的测试配置
        
        Args:
            env: 环境类型 (local/server)
            dataset: 数据集名称 (twitter2015/twitter2017/mix)
            task_name: 任务名称 (mate/mner/mabsa)
            mode: 模态类型 (text_only/multimodal)
            use_label_embedding: 是否使用标签嵌入
            output_suffix: 输出文件后缀
            kaggle_mode: 是否为Kaggle环境
        
        Returns:
            配置字典
        """
        # 使用基础生成器生成配置
        config = self.base_generator.generate_task_sequence_config(
            env=env,
            dataset=dataset,
            task_sequence=[task_name],  # 单任务
            mode_sequence=[mode],
            strategy="none",            # 无持续学习策略
            use_label_embedding=use_label_embedding,
            seq_suffix=output_suffix,
            **self.recommended_hyperparams
        )
        
        # 添加CRF和Span Loss配置到每个任务
        ablation_config = self.ablation_configs[ablation_type]
        for task in config["tasks"]:
            # 只对序列任务启用CRF
            if task["task_name"] in ["mate", "mner", "mabsa"]:
                task.update({
                    "use_crf": ablation_config["use_crf"],
                    "use_span_loss": ablation_config["use_span_loss"],
                    "boundary_weight": ablation_config["boundary_weight"],
                    "span_f1_weight": ablation_config["span_f1_weight"],
                    "transition_weight": ablation_config["transition_weight"]
                })
                
                # 根据任务调整num_labels（确保正确）
                if task["task_name"] == "mate":
                    task["num_labels"] = 3   # O, B, I
                elif task["task_name"] == "mner":
                    task["num_labels"] = 9   # O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
                elif task["task_name"] == "mabsa":
                    task["num_labels"] = 7   # O, B-NEG, I-NEG, B-NEU, I-NEU, B-POS, I-POS
        
        # 添加测试元信息
        config["test_info"] = {
            "purpose": "Ablation study for CRF and Span Loss",
            "ablation_type": ablation_type,
            "configuration": ablation_config["description"],
            "fixes_included": [
                "CRF layer for BIO constraints" if ablation_config["use_crf"] else "No CRF",
                "valid_len bug fix",
                "Span-level boundary loss" if ablation_config["use_span_loss"] else "No Span Loss"
            ],
            "expected_improvement": self._get_expected_improvement(ablation_type)
        }
        
        # Kaggle特殊配置
        if kaggle_mode:
            config["kaggle_mode"] = True
            config["kaggle_output_path"] = "/kaggle/working"
            
            # 修改输出路径到Kaggle可写目录
            original_output = config["global_params"]["output_model_path"]
            config["global_params"]["output_model_path"] = f"/kaggle/working/{Path(original_output).name}"
            
            original_train_info = config["global_params"]["train_info_json"]
            config["global_params"]["train_info_json"] = f"/kaggle/working/{Path(original_train_info).name}"
        
        return config
    
    def _get_expected_improvement(self, ablation_type: str) -> Dict[str, str]:
        """获取不同配置的预期提升"""
        improvements = {
            "baseline": {
                "chunk_f1": "~32% (baseline)",
                "boundary_detection": "~45% (baseline)",
                "illegal_sequences": "Yes (possible)"
            },
            "crf_only": {
                "chunk_f1": "+35-40% vs baseline",
                "boundary_detection": "+25-30% vs baseline",
                "illegal_sequences": "No (CRF enforces constraints)"
            },
            "span_only": {
                "chunk_f1": "+30-35% vs baseline",
                "boundary_detection": "+35-40% vs baseline",
                "illegal_sequences": "Yes (no CRF)"
            },
            "crf_and_span": {
                "chunk_f1": "+40-45% vs baseline (best)",
                "boundary_detection": "+40-45% vs baseline (best)",
                "illegal_sequences": "No (CRF enforces constraints)"
            }
        }
        return improvements.get(ablation_type, {})
    
    def generate_all_test_configs(self,
                                  env: str = "server",
                                  dataset: str = "twitter2015",
                                  output_dir: str = "scripts/configs/crf_test",
                                  kaggle_mode: bool = False):
        """生成所有测试配置"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("CRF Ablation Study配置生成器")
        print("="*80)
        print(f"\n生成配置到: {output_path}")
        print(f"环境: {env}")
        print(f"数据集: {dataset}")
        print(f"Kaggle模式: {kaggle_mode}")
        print(f"\n测试任务: {', '.join(self.tasks)}")
        print(f"Ablation配置: {len(self.ablation_configs)}")
        print(f"总配置数: {len(self.tasks) * len(self.ablation_configs)}")
        print("\nAblation类型:")
        for ablation_name, config in self.ablation_configs.items():
            print(f"  {ablation_name}: {config['description']}")
        print("\n超参数:")
        for key, value in self.recommended_hyperparams.items():
            print(f"  {key}: {value}")
        print("="*80 + "\n")
        
        configs_generated = []
        
        for task_name in self.tasks:
            print(f"\n生成任务: {task_name.upper()}")
            
            for ablation_type, ablation_config in self.ablation_configs.items():
                # 配置文件名
                if kaggle_mode:
                    config_name = f"kaggle_{ablation_type}_{dataset}_{task_name}.json"
                else:
                    config_name = f"{ablation_type}_{dataset}_{task_name}.json"
                
                config_file = output_path / config_name
                
                # 生成配置
                config = self.generate_single_task_config(
                    env=env,
                    dataset=dataset,
                    task_name=task_name,
                    mode="multimodal",
                    use_label_embedding=False,
                    output_suffix=f"_{ablation_type}",
                    kaggle_mode=kaggle_mode,
                    ablation_type=ablation_type
                )
                
                # 保存配置
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                configs_generated.append({
                    "file": config_file.name,
                    "path": config_file.as_posix(),
                    "task": task_name,
                    "dataset": dataset,
                    "ablation_type": ablation_type,
                    "mode": "multimodal"
                })
                
                print(f"  ✓ [{ablation_type}] {config_name}")
        
        # 生成索引文件
        index_file = output_path / "ablation_study_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({
                "description": "CRF and Span Loss Ablation Study",
                "purpose": "比较4种配置的效果: baseline, CRF only, Span Loss only, CRF+Span Loss",
                "total_configs": len(configs_generated),
                "ablation_configs": len(self.ablation_configs),
                "tasks": self.tasks,
                "dataset": dataset,
                "hyperparameters": self.recommended_hyperparams,
                "ablation_types": {
                    name: config["description"]
                    for name, config in self.ablation_configs.items()
                },
                "kaggle_mode": kaggle_mode,
                "configs": configs_generated
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 索引文件: {index_file}")
        print(f"\n✅ 总共生成 {len(configs_generated)} 个配置文件")
        print(f"   - {len(self.tasks)} 任务 × {len(self.ablation_configs)} Ablation配置\n")
        
        return configs_generated
    
    def generate_batch_runner(self, 
                              configs: list,
                              output_dir: str,
                              kaggle_mode: bool = False):
        """生成批量运行脚本"""
        output_path = Path(output_dir)
        
        if kaggle_mode:
            script_name = "run_kaggle_crf_tests.py"
        else:
            script_name = "run_crf_tests.sh"
        
        script_file = output_path / script_name
        
        if kaggle_mode:
            # Kaggle Python运行脚本
            script_content = '''#!/usr/bin/env python3
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
    
    print(f"\\nCompleted in {elapsed/60:.1f} minutes")
    print(f"Exit code: {result.returncode}\\n")
    
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
    print(f"\\n总实验数: {index['total_configs']}")
    print(f"任务: {', '.join(index['tasks'])}")
    print("\\n")
    
    results = []
    for i, config_info in enumerate(index['configs'], 1):
        config_file = f"scripts/configs/crf_test/{config_info['file']}"
        task = config_info['task']
        
        print(f"\\n[{i}/{index['total_configs']}] {task.upper()}")
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
    print("\\n" + "="*80)
    print("测试完成!")
    print("="*80)
    success_count = sum(1 for r in results if r['success'])
    print(f"\\n成功: {success_count}/{len(results)}")
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"  {status} {r['task']}")
    
    print("\\n结果保存在: /kaggle/working/")
    print("  - checkpoints/: 模型文件")
    print("  - train_info*.json: 训练详情")
    print("  - test_progress.json: 测试进度")

if __name__ == "__main__":
    main()
'''
        else:
            # 本地Shell脚本
            lines = [
                "#!/bin/bash",
                "# CRF修复测试批量运行脚本",
                "",
                "echo '=========================================='",
                "echo 'CRF修复测试 - 批量运行'",
                "echo '=========================================='",
                f"echo '总实验数: {len(configs)}'",
                "echo ''",
                ""
            ]
            
            for i, config_info in enumerate(configs, 1):
                task = config_info['task']
                config_file = config_info['path']
                lines.extend([
                    f"echo '[{i}/{len(configs)}] 运行任务: {task.upper()}'",
                    f"python -m scripts.train_with_zero_shot --config {config_file}",
                    "if [ $? -ne 0 ]; then",
                    f"    echo '✗ {task} 失败'",
                    "else",
                    f"    echo '✓ {task} 完成'",
                    "fi",
                    "echo ''",
                    ""
                ])
            
            lines.extend([
                "echo '=========================================='",
                "echo '所有测试完成!'",
                "echo '=========================================='",
                ""
            ])
            
            script_content = "\n".join(lines)
        
        # 写入脚本
        with open(script_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write(script_content)
        
        # 添加执行权限（Unix系统）
        import os
        import stat
        if not kaggle_mode:
            os.chmod(script_file, os.stat(script_file).st_mode | stat.S_IEXEC)
        
        print(f"✓ 批量运行脚本: {script_file}")
        if not kaggle_mode:
            print(f"  使用方法: ./{script_file.name}")
        else:
            print(f"  使用方法: python {script_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="生成CRF修复测试配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成本地测试配置
  python scripts/generate_crf_test_configs.py
  
  # 生成Kaggle测试配置
  python scripts/generate_crf_test_configs.py --kaggle
  
  # 指定数据集
  python scripts/generate_crf_test_configs.py --dataset twitter2017
  
生成的文件:
  - crf_test_{dataset}_{task}.json: 任务配置
  - test_index.json: 配置索引
  - run_crf_tests.sh/py: 批量运行脚本
        """
    )
    
    parser.add_argument("--env", type=str, default="server",
                       choices=["local", "server"],
                       help="环境类型 (default: server)")
    
    parser.add_argument("--dataset", type=str, default="twitter2015",
                       choices=["twitter2015", "twitter2017", "mix"],
                       help="数据集名称 (default: twitter2015)")
    
    parser.add_argument("--output_dir", type=str,
                       default="scripts/configs/crf_test",
                       help="输出目录 (default: scripts/configs/crf_test)")
    
    parser.add_argument("--kaggle", action="store_true",
                       help="生成Kaggle环境配置")
    
    args = parser.parse_args()
    
    # 生成配置
    generator = CRFTestConfigGenerator()
    configs = generator.generate_all_test_configs(
        env=args.env,
        dataset=args.dataset,
        output_dir=args.output_dir,
        kaggle_mode=args.kaggle
    )
    
    # 生成批量运行脚本
    generator.generate_batch_runner(configs, args.output_dir, args.kaggle)
    
    # 打印使用说明
    print("="*80)
    print("使用说明")
    print("="*80)
    print("\n1. 查看配置:")
    print(f"   cat {args.output_dir}/test_index.json")
    
    print("\n2. 运行单个实验:")
    print(f"   python -m scripts.train_with_zero_shot \\")
    print(f"     --config {args.output_dir}/crf_test_{args.dataset}_mate.json")
    
    print("\n3. 批量运行所有实验:")
    if args.kaggle:
        print(f"   python {args.output_dir}/run_kaggle_crf_tests.py")
    else:
        print(f"   ./{args.output_dir}/run_crf_tests.sh")
    
    print("\n4. 检查结果:")
    print("   - 训练日志中查找:")
    print("     [MATE] Head initialized with CRF")
    print("     ✓ Span Loss enabled")
    print("   - 评估指标中查找:")
    print("     Chunk F1: XX.XX% (主指标1)")
    
    print("\n预期改进:")
    print("  - Chunk F1: 30% → 60-75% (+30-45%)")
    print("  - 边界检测: +20-30%")
    print("="*80)


if __name__ == "__main__":
    main()

