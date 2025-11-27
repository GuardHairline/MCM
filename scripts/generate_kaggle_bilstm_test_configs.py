#!/usr/bin/env python3
"""
生成BiLSTM Head测试配置文件 - Kaggle多账号版本

目的：测试新实现的BiLSTM-CRF任务头的效果和不同超参数配置

实验设计：
- 3个任务: MATE, MNER, MABSA
- 每个任务测试3种BiLSTM配置
- 每个配置: text_only session → multimodal session（持续学习序列）
- 总计9个配置，18个训练session

账号分配策略：
- Account 1: MATE (config_small, config_default, config_large)
- Account 2: MNER (config_small, config_default, config_large)  
- Account 3: MABSA (config_small, config_default, config_large)

时间估算（每个配置包含2个session）：
- Twitter2015: ~3-3.7小时/配置（text_only ~1.5h + multimodal ~1.5-2h）
- 3个配置 = 9-11小时
- 总时间在12小时Kaggle限制内
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_task_config import TaskConfigGenerator

class KaggleBiLSTMTestGenerator:
    """Kaggle多账号BiLSTM测试配置生成器"""
    
    def __init__(self, keep_checkpoints: bool = True, use_bilstm: int = 1, enable_bilstm_head: int = 1):
        # 使用基础配置生成器
        self.base_generator = TaskConfigGenerator()
        self.keep_checkpoints = keep_checkpoints
        self.use_bilstm = use_bilstm
        self.enable_bilstm_head = enable_bilstm_head
        
        # BiLSTM超参数配置
        self.bilstm_configs = {
            # "config_small": {
            #     "description": "小型配置：hidden_size=128, num_layers=1",
            #     "use_bilstm": 1,
            #     "bilstm_hidden_size": 128,
            #     "bilstm_num_layers": 1,
            #     "lstm_lr": 1e-4,
            #     "crf_lr": 1e-3,
            # },
            "config_default": {
                "description": "默认配置：hidden_size=256, num_layers=2",
                "use_bilstm": 1,
                "bilstm_hidden_size": 256,
                "bilstm_num_layers": 2,
                "lstm_lr": 1e-4,
                "crf_lr": 1e-3,
            },
            # "config_large": {
            #     "description": "大型配置：hidden_size=512, num_layers=2",
            #     "use_bilstm": 1,
            #     "bilstm_hidden_size": 512,
            #     "bilstm_num_layers": 2,
            #     "lstm_lr": 5e-5,  # 更大模型使用更小学习率
            #     "crf_lr": 5e-4,
            # },
        }
        
        # 3个账号的实验分配
        self.account_assignments = {
            "account_1": {
                "name": "Account 1 - MATE BiLSTM测试",
                "task": "mate",
                "configs": [ "config_default"],
                "description": "MATE任务：测试三种BiLSTM配置"
            },
            "account_2": {
                "name": "Account 2 - MNER BiLSTM测试",
                "task": "mner",
                "configs": ["config_default"],
                "description": "MNER任务：测试三种BiLSTM配置"
            },
            "account_3": {
                "name": "Account 3 - MABSA BiLSTM测试",
                "task": "mabsa",
                "configs": ["config_default"],
                "description": "MABSA任务：测试三种BiLSTM配置"
            }
        }
        
        # 时间估算（分钟）- 每个配置包含 text_only + multimodal 两个session
        self.time_estimates = {
            "mate": 180,     # 3小时/配置
            "mner": 200,     # 3.3小时/配置
            "mabsa": 220     # 3.7小时/配置
        }
        
        # 任务标签数量
        self.task_num_labels = {
            "mate": 3,
            "mner": 9,
            "mabsa": 7
        }
    
    def generate_task_config(self, task: str, mode: str, config_type: str, 
                            dataset: str = "twitter2015") -> Dict[str, Any]:
        """生成单个任务配置 - 使用基础生成器确保参数完整"""
        
        # 使用基础生成器生成完整配置
        # 注意：这会生成一个包含tasks列表的完整配置，我们需要提取其中的任务配置
        full_config = self.base_generator.generate_task_sequence_config(
            env="server",  # 使用server环境，后面会调整路径
            dataset=dataset,
            task_sequence=[task],
            mode_sequence=[mode],
            strategy="none",  # 单任务，无持续学习策略
            use_label_embedding=False,
            seq_suffix=f"_{mode}_{config_type}",
            # 训练超参数（修复：与simple_ner_training.py对齐）
            lr=1e-5,            # ✅ 降低到1e-5（原2e-5过高）
            step_size=10,       # ✅ 改为10轮衰减一次（更温和，原5轮太激进）
            gamma=0.5,
            epochs=20,          # ✅ 增加到20轮（原15轮不够）
            patience=999,       # ✅ 禁用早停（远大于epochs），只记录最优但完整训练20轮
            batch_size=16,
            weight_decay=0.01
        )
        
        # 提取第一个任务的配置（因为我们只生成了一个任务）
        if full_config["tasks"]:
            task_config = full_config["tasks"][0]
        else:
            raise ValueError("基础生成器未能生成任务配置")
        
        # 添加BiLSTM特定配置
        bilstm_params = self.bilstm_configs[config_type]
        task_config.update(bilstm_params)
        task_config["use_bilstm"] = self.use_bilstm
        task_config["enable_bilstm_head"] = self.enable_bilstm_head
        task_config["debug_samples"] = 100  # 便于训练后输出dev样本对齐日志
        # 确保CRF启用
        task_config["use_crf"] = 1
        task_config["use_span_loss"] = 0  # 不使用span loss，只测试BiLSTM-CRF
        
        # 确保triaffine关闭
        task_config["triaffine"] = 0
        
        return task_config
    
    def generate_account_configs(self,
                                 account_id: str,
                                 dataset: str = "twitter2015",
                                 output_dir: str = "scripts/configs/kaggle_bilstm_test"):
        """为指定账号生成配置"""
        
        if account_id not in self.account_assignments:
            raise ValueError(f"Unknown account: {account_id}. Available: {list(self.account_assignments.keys())}")
        
        assignment = self.account_assignments[account_id]
        task = assignment["task"]
        config_types = assignment["configs"]
        
        output_path = Path(output_dir) / account_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        configs = []
        
        print(f"\n{'='*80}")
        print(f"{assignment['name']}")
        print(f"{'='*80}")
        print(f"任务: {task.upper()}")
        print(f"配置: {', '.join(config_types)}")
        print(f"描述: {assignment['description']}")
        print(f"预计时间: {len(config_types) * self.time_estimates[task]} 分钟")
        print(f"{'='*80}\n")
        
        for config_type in config_types:
            config_name = f"kaggle_bilstm_{config_type}_{dataset}_{task}.json"
            config_file = output_path / config_name
            
            # 生成配置 - 包含 text_only 和 multimodal 两个session（持续学习序列）
            tasks_list = []
            
            # Session 1: text_only
            task1 = self.generate_task_config(task, "text_only", config_type, dataset)
            # ✅ 修复：手动设置正确的session_name
            task1["session_name"] = f"{task}_1"  # 第1个session
            # 第一个任务的输入输出路径
            if "pretrained_model_path" not in task1 or task1["pretrained_model_path"]:
                task1["pretrained_model_path"] = None  # 从头开始
            task1["output_model_path"] = f"checkpoints/kaggle_{task}_{dataset}_textonly_{config_type}.pt"
            tasks_list.append(task1)
            
            # Session 2: multimodal（持续学习，使用text_only的输出作为输入）
            task2 = self.generate_task_config(task, "multimodal", config_type, dataset)
            # ✅ 修复：手动设置正确的session_name
            task2["session_name"] = f"{task}_2"  # 第2个session（不同于第1个！）
            task2["pretrained_model_path"] = task1["output_model_path"]
            task2["output_model_path"] = f"checkpoints/kaggle_{task}_{dataset}_multimodal_{config_type}.pt"
            tasks_list.append(task2)
            
            # 全局参数
            global_params = {
                "train_info_json": f"checkpoints/train_info_kaggle_{task}_{dataset}_{config_type}.json",
                "output_model_path": f"checkpoints/kaggle_{task}_{dataset}_final_{config_type}.pt",
                "ewc_dir": "checkpoints/ewc",
                "gem_mem_dir": "checkpoints/gem_memory",
                "description": f"BiLSTM {config_type} test for {task} on {dataset}",
                "save_checkpoints": 1 if self.keep_checkpoints else 0,
                "enable_bilstm_head": self.enable_bilstm_head
            }
            
            # 完整配置
            full_config = {
                "tasks": tasks_list,
                "global_params": global_params,
                "experiment_info": {
                    "purpose": "BiLSTM Head Parameter Testing",
                    "config_type": config_type,
                    "configuration": self.bilstm_configs[config_type]["description"],
                    "task": task,
                    "dataset": dataset,
                    "mode_sequence": ["text_only", "multimodal"],
                    "notes": "测试新实现的BiLSTM-CRF任务头"
                },
                "kaggle_mode": True,
                "kaggle_output_path": "/kaggle/working"
            }
            
            # 保存配置
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, indent=2, ensure_ascii=False)
            
            configs.append({
                "file": config_name,
                "path": config_file.as_posix(),
                "task": task,
                "config_type": config_type,
                "dataset": dataset
            })
            
            print(f"  OK Generated: {config_name}")
        
        # 生成账号索引文件
        index_file = output_path / f"{account_id}_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({
                "account_id": account_id,
                "account_name": assignment["name"],
                "task": task,
                "config_types": config_types,
                "description": assignment["description"],
                "estimated_time_minutes": len(config_types) * self.time_estimates[task],
                "total_configs": len(configs),
                "configs": configs
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n  OK Index file: {index_file}\n")
        
        return configs, output_path
    
    def generate_all_accounts(self,
                             dataset: str = "twitter2015",
                             output_dir: str = "scripts/configs/kaggle_bilstm_test"):
        """为所有3个账号生成配置"""
        
        all_configs = {}
        
        for account_id in self.account_assignments.keys():
            configs, account_dir = self.generate_account_configs(
                account_id=account_id,
                dataset=dataset,
                output_dir=output_dir
            )
            all_configs[account_id] = {
                "configs": configs,
                "directory": str(account_dir)
            }
            
            # 为每个账号生成运行脚本
            self._generate_account_runner(account_id, configs, account_dir)
        
        # 生成总索引文件
        master_index = Path(output_dir) / "master_index.json"
        with open(master_index, 'w', encoding='utf-8') as f:
            json.dump({
                "description": "3账号BiLSTM测试总索引",
                "purpose": "测试新实现的BiLSTM-CRF任务头",
                "total_accounts": len(self.account_assignments),
                "total_configs": sum(len(v["configs"]) for v in all_configs.values()),
                "accounts": {
                    acc_id: {
                        "name": self.account_assignments[acc_id]["name"],
                        "task": self.account_assignments[acc_id]["task"],
                        "config_types": self.account_assignments[acc_id]["configs"],
                        "configs_count": len(all_configs[acc_id]["configs"]),
                        "directory": all_configs[acc_id]["directory"]
                    }
                    for acc_id in self.account_assignments.keys()
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"OK All account configs generated successfully")
        print(f"{'='*80}")
        print(f"总账号数: {len(self.account_assignments)}")
        print(f"总配置数: {sum(len(v['configs']) for v in all_configs.values())}")
        print(f"输出目录: {output_dir}")
        print(f"总索引文件: {master_index}")
        print(f"{'='*80}\n")
        
        # 生成部署指南
        self._generate_deployment_guide(Path(output_dir))
        
        # 生成结果分析脚本
        self._generate_analysis_script(Path(output_dir), all_configs)
        
        return all_configs
    
    def _generate_account_runner(self, account_id: str, configs: list, output_dir: Path):
        """为单个账号生成Kaggle运行脚本"""
        
        assignment = self.account_assignments[account_id]
        runner_path = output_dir / f"run_{account_id}.py"
        
        with open(runner_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(f'''#!/usr/bin/env python3
"""
Kaggle运行脚本 - {assignment["name"]}

此脚本在Kaggle Notebook中运行
任务: {assignment["task"].upper()}
配置: {", ".join(assignment["configs"])}
预计时间: {len(configs) * self.time_estimates[assignment["task"]]} 分钟

使用说明:
1. 在Kaggle Notebook中创建新的Code
2. 设置加速器为 GPU T4 或 P100
3. 添加数据集: mcm-project (包含代码和数据)
4. 复制此脚本内容到Notebook
5. 点击 Run All

测试目的:
- 验证新实现的BiLSTM-CRF任务头功能正确
- 比较不同BiLSTM超参数配置的效果
- 确认text_only → multimodal持续学习流程正常
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
print("{assignment['name']}")
print("="*80)
print(f"任务: {assignment['task'].upper()}")
print(f"配置数: {len(configs)}")
print(f"预计时间: {len(configs) * self.time_estimates[assignment['task']]} 分钟")
print("="*80 + "\\n")

# ============================================================================
# Step 1: 项目设置
# ============================================================================

def setup_project():
    """设置项目环境"""
    print("\\n" + "="*80)
    print("Step 1: 设置项目环境")
    print("="*80)
    
    project_src = Path(KAGGLE_INPUT) / PROJECT_DATASET
    project_dst = Path(KAGGLE_WORKING) / "MCM"
    
    # 检查源目录
    if not project_src.exists():
        print(f"❌ 错误: 未找到项目数据集 {{project_src}}")
        print("   请确保已添加 mcm-project 数据集")
        sys.exit(1)
    
    # 复制项目文件
    if project_dst.exists():
        print(f"项目目录已存在，删除旧版本...")
        shutil.rmtree(project_dst)
    
    print(f"复制项目文件...")
    shutil.copytree(project_src, project_dst)
    
    # 切换到项目目录
    os.chdir(project_dst)
    sys.path.insert(0, str(project_dst))
    
    print(f"✓ 项目目录: {{project_dst}}")
    print(f"✓ 当前工作目录: {{os.getcwd()}}")
    
    return project_dst

# ============================================================================
# Step 2: 运行实验
# ============================================================================

def run_experiments(project_dir: Path):
    """运行所有实验"""
    print("\\n" + "="*80)
    print("Step 2: 运行BiLSTM测试实验")
    print("="*80)
    
    configs = {json.dumps([{"file": c["file"], "task": c["task"], "config_type": c["config_type"]} for c in configs])}
    
    results = {{}}
    
    for idx, config_info in enumerate(configs, 1):
        config_file = config_info["file"]
        config_path = project_dir / "scripts" / "configs" / "kaggle_bilstm_test" / "{account_id}" / config_file
        
        print(f"\\n{{'='*80}}")
        print(f"实验 {{idx}}/{{len(configs)}}: {{config_info['config_type']}}")
        print(f"{{'='*80}}")
        print(f"任务: {{config_info['task'].upper()}}")
        print(f"配置文件: {{config_file}}")
        
        start_time = time.time()
        
        try:
            # 运行训练
            cmd = [
                sys.executable, "-m", "scripts.train_with_zero_shot",
                "--config", str(config_path)
            ]
            
            print(f"\\n执行命令: {{' '.join(cmd)}}\\n")
            
            result = subprocess.run(
                cmd,
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=7200  # 2小时超时
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"\\n✓ 实验完成，耗时: {{elapsed/60:.1f}} 分钟")
                results[config_file] = {{
                    "status": "success",
                    "elapsed_minutes": elapsed / 60,
                    "config_type": config_info["config_type"]
                }}
            else:
                print(f"\\n❌ 实验失败")
                print("STDERR:", result.stderr[-500:] if result.stderr else "")
                results[config_file] = {{
                    "status": "failed",
                    "error": result.stderr[-200:] if result.stderr else "Unknown error"
                }}
        
        except subprocess.TimeoutExpired:
            print(f"\\n❌ 实验超时（2小时）")
            results[config_file] = {{
                "status": "timeout"
            }}
        except Exception as e:
            print(f"\\n❌ 实验异常: {{e}}")
            results[config_file] = {{
                "status": "error",
                "error": str(e)
            }}
    
    return results

# ============================================================================
# Step 3: 收集和导出结果
# ============================================================================

def collect_results(project_dir: Path, results: dict):
    """收集结果文件"""
    print("\\n" + "="*80)
    print("Step 3: 收集结果")
    print("="*80)
    
    # 创建结果目录
    results_dir = Path(KAGGLE_WORKING) / "results_{account_id}"
    results_dir.mkdir(exist_ok=True)
    
    # 复制训练信息文件
    checkpoint_dir = project_dir / "checkpoints"
    if checkpoint_dir.exists():
        for train_info_file in checkpoint_dir.glob("train_info_*.json"):
            shutil.copy(train_info_file, results_dir)
            print(f"✓ {{train_info_file.name}}")
        
        # 复制图片文件
        for img_file in checkpoint_dir.glob("*.png"):
            shutil.copy(img_file, results_dir)
            print(f"✓ {{img_file.name}}")
    
    # 保存运行摘要
    summary_file = results_dir / "run_summary_{account_id}.json"
    with open(summary_file, 'w') as f:
        json.dump({{
            "account_id": "{account_id}",
            "task": "{assignment['task']}",
            "total_configs": len(results),
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }}, f, indent=2)
    
    print(f"\\n✓ 结果已保存到: {{results_dir}}")
    print(f"✓ 运行摘要: {{summary_file}}")
    
    return results_dir

# ============================================================================
# 主流程
# ============================================================================

if __name__ == "__main__":
    try:
        # Step 1: 设置项目
        project_dir = setup_project()
        
        # Step 2: 运行实验
        results = run_experiments(project_dir)
        
        # Step 3: 收集结果
        results_dir = collect_results(project_dir, results)
        
        # 打印最终摘要
        print("\\n" + "="*80)
        print("✅ 所有实验完成")
        print("="*80)
        success_count = sum(1 for r in results.values() if r.get("status") == "success")
        print(f"成功: {{success_count}}/{{len(results)}}")
        print(f"结果目录: {{results_dir}}")
        print("="*80)
        
    except Exception as e:
        print(f"\\n❌ 运行失败: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
''')
        
        print(f"  OK Runner script: {runner_path}")
    
    def _generate_deployment_guide(self, output_dir: Path):
        """生成部署指南"""
        guide_path = output_dir / "KAGGLE_DEPLOYMENT_GUIDE.md"
        
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write('''# Kaggle BiLSTM测试部署指南

## 实验目的

测试新实现的BiLSTM-CRF任务头在MATE、MNER、MABSA三个任务上的表现，并比较不同超参数配置的效果。

## 实验设计

### 任务分配

| 账号 | 任务 | 配置 | 预计时间 |
|------|------|------|----------|
| Account 1 | MATE | small, default, large | ~9小时 |
| Account 2 | MNER | small, default, large | ~10小时 |
| Account 3 | MABSA | small, default, large | ~11小时 |

### BiLSTM配置

1. **config_small**: 
   - hidden_size=128, num_layers=1
   - 快速训练，基线对比

2. **config_default**: 
   - hidden_size=256, num_layers=2
   - 推荐的默认配置

3. **config_large**: 
   - hidden_size=512, num_layers=2
   - 更大容量，可能效果更好但训练慢

### 持续学习序列

每个配置包含2个session的持续学习序列：
1. **text_only**: 仅使用文本模态
2. **multimodal**: 使用文本+图像（继承text_only的权重）

## 部署步骤

### 1. 准备Kaggle数据集

确保已上传 `mcm-project` 数据集，包含：
- 完整项目代码
- 数据文件 (Twitter2015)
- 预训练模型

### 2. 为每个账号创建Notebook

对每个账号 (account_1, account_2, account_3)：

1. 登录对应的Kaggle账号
2. 创建新Notebook
3. 设置：
   - Accelerator: GPU T4 或 P100
   - Internet: On (如需下载预训练模型)
4. 添加数据集: `mcm-project`
5. 复制对应的 `run_account_X.py` 内容到Notebook
6. 点击 "Run All"

### 3. 监控执行

- 每个账号运行约9-11小时
- 关注训练日志，确认没有错误
- 检查GPU使用率

### 4. 导出结果

完成后，从 `/kaggle/working/results_account_X/` 下载：
- `train_info_*.json`: 训练信息
- `*.png`: 热力图
- `run_summary_*.json`: 运行摘要

## 预期结果

### 关键指标

- **Chunk F1 (Span-level)**: 序列任务的主要评估指标
- **Token Micro F1**: token级别的F1分数
- **Training Time**: 训练时间对比

### 对比维度

1. **任务间对比**: MATE vs MNER vs MABSA
2. **配置间对比**: small vs default vs large
3. **模态对比**: text_only vs multimodal
4. **持续学习效果**: 从text_only到multimodal的迁移

## 结果分析

运行 `analyze_bilstm_results.py` 自动生成：
- 性能对比表格
- 训练时间分析
- 超参数影响分析
- 最佳配置推荐

## 故障排除

### 常见问题

1. **OOM (Out of Memory)**
   - 减小 batch_size
   - 使用 config_small

2. **训练时间过长**
   - 减少 epochs
   - 只运行部分配置

3. **数据集未找到**
   - 确认 mcm-project 数据集已添加
   - 检查数据集版本

## 下一步

1. 收集所有账号的结果
2. 运行分析脚本
3. 根据结果选择最佳配置
4. 在完整数据集上进行完整实验
''')
        
        print(f"  OK Deployment guide: {guide_path}")
    
    def _generate_analysis_script(self, output_dir: Path, all_configs: dict):
        """生成结果分析脚本"""
        script_path = output_dir / "analyze_bilstm_results.py"
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('''#!/usr/bin/env python3
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
        f.write("# BiLSTM测试结果分析报告\\n\\n")
        
        # 按任务分组
        f.write("## 按任务分析\\n\\n")
        for task in df["task"].unique():
            task_df = df[df["task"] == task]
            f.write(f"### {task.upper()}\\n\\n")
            f.write(task_df.to_markdown(index=False))
            f.write("\\n\\n")
        
        # 按配置分组
        f.write("## 按配置分析\\n\\n")
        for config_type in df["config_type"].unique():
            config_df = df[df["config_type"] == config_type]
            f.write(f"### {config_type}\\n\\n")
            f.write(config_df.to_markdown(index=False))
            f.write("\\n\\n")
        
        # 最佳配置
        f.write("## 最佳配置推荐\\n\\n")
        best_by_task = df.loc[df.groupby("task")["test_chunk_f1"].idxmax()]
        f.write(best_by_task[["task", "config_type", "mode", "test_chunk_f1"]].to_markdown(index=False))
        f.write("\\n")
    
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
    
    print("\\n✅ 分析完成")
''')
        
        print(f"  OK Analysis script: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="生成Kaggle BiLSTM测试配置")
    
    parser.add_argument("--account", type=str, choices=["account_1", "account_2", "account_3", "all"],
                       default="all", help="生成哪个账号的配置")
    parser.add_argument("--dataset", type=str, default="twitter2015",
                       help="数据集名称")
    parser.add_argument("--output-dir", type=str, default="scripts/configs/kaggle_bilstm_test",
                       help="输出目录")
    parser.add_argument("--keep-checkpoints", type=int, choices=[0, 1], default=1,
                       help="是否在训练后保留checkpoint（1=保留，0=清理）")
    parser.add_argument("--use-bilstm", type=int, choices=[0, 1], default=1,
                       help="任务配置中use_bilstm的值")
    parser.add_argument("--enable-bilstm-head", type=int, choices=[0, 1], default=1,
                       help="任务配置中enable_bilstm_head的值")
    
    args = parser.parse_args()
    
    generator = KaggleBiLSTMTestGenerator(
        keep_checkpoints=bool(args.keep_checkpoints),
        use_bilstm=args.use_bilstm,
        enable_bilstm_head=args.enable_bilstm_head
    )
    
    if args.account == "all":
        generator.generate_all_accounts(
            dataset=args.dataset,
            output_dir=args.output_dir
        )
    else:
        generator.generate_account_configs(
            account_id=args.account,
            dataset=args.dataset,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()

