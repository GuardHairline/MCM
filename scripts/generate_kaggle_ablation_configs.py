#!/usr/bin/env python3
"""
生成CRF和Span Loss Ablation Study配置文件 - Kaggle多账号版本

特性：
1. 为6个Kaggle账号分配不同的实验组合
2. 每个账号2个配置，每个配置包含text_only + multimodal两个session
3. 自动生成账号专属运行脚本
4. 支持结果汇总和分析

实验设计：
- 3个任务: MATE, MNER, MABSA
- 4种配置: baseline, crf_only, span_only, crf_and_span
- 每个配置: text_only session → multimodal session（持续学习序列）
- 总计12个配置，24个训练session

账号分配策略：
- Account 1: MATE (baseline + crf_and_span)
- Account 2: MATE (crf_only + span_only)  
- Account 3: MNER (baseline + crf_and_span)
- Account 4: MNER (crf_only + span_only)
- Account 5: MABSA (baseline + crf_and_span)
- Account 6: MABSA (crf_only + span_only)

时间估算（每个配置包含2个session）：
- Twitter2015: ~3-3.7小时/配置（text_only ~1.5h + multimodal ~1.5-2h）
- 2个配置 = 6-7.4小时
- 留余量：8小时内完成（远低于12小时限制）
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_crf_test_configs import CRFTestConfigGenerator


class KaggleAblationStudyGenerator:
    """Kaggle多账号Ablation Study配置生成器"""
    
    def __init__(self):
        self.base_generator = CRFTestConfigGenerator()
        
        # 3个账号的实验分配（简化版：只对比 Baseline vs CRF Only）
        # 移除 Span Loss（有严重问题）
        self.account_assignments = {
            "account_1": {
                "name": "Account 1 - MATE (Baseline vs CRF)",
                "task": "mate",
                "ablations": ["baseline", "crf_only"],
                "description": "MATE任务：对比baseline和CRF效果"
            },
            "account_2": {
                "name": "Account 2 - MNER (Baseline vs CRF)",
                "task": "mner",
                "ablations": ["baseline", "crf_only"],
                "description": "MNER任务：对比baseline和CRF效果"
            },
            "account_3": {
                "name": "Account 3 - MABSA (Baseline vs CRF)",
                "task": "mabsa",
                "ablations": ["baseline", "crf_only"],
                "description": "MABSA任务：对比baseline和CRF效果"
            }
        }
        
        # 时间估算（分钟）- 每个配置包含 text_only + multimodal 两个session
        self.time_estimates = {
            "mate": 180,     # 3小时/配置 (text_only ~1.5h + multimodal ~1.5h)
            "mner": 200,     # 3.3小时/配置 (text_only ~1.7h + multimodal ~1.6h)
            "mabsa": 220     # 3.7小时/配置 (text_only ~1.8h + multimodal ~1.9h)
        }
    
    def generate_account_configs(self,
                                 account_id: str,
                                 env: str = "server",
                                 dataset: str = "twitter2015",
                                 output_dir: str = "scripts/configs/kaggle_ablation"):
        """为指定账号生成配置"""
        
        if account_id not in self.account_assignments:
            raise ValueError(f"Unknown account: {account_id}. Available: {list(self.account_assignments.keys())}")
        
        assignment = self.account_assignments[account_id]
        task = assignment["task"]
        ablations = assignment["ablations"]
        
        output_path = Path(output_dir) / account_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        configs = []
        
        print(f"\n{'='*80}")
        print(f"{assignment['name']}")
        print(f"{'='*80}")
        print(f"任务: {task.upper()}")
        print(f"配置: {', '.join(ablations)}")
        print(f"描述: {assignment['description']}")
        print(f"预计时间: {len(ablations) * self.time_estimates[task]} 分钟")
        print(f"{'='*80}\n")
        
        for ablation_type in ablations:
            config_name = f"kaggle_{ablation_type}_{dataset}_{task}.json"
            config_file = output_path / config_name
            
            # 生成配置 - 包含 text_only 和 multimodal 两个session
            # 使用 generate_task_sequence_config 直接生成持续学习序列
            config = self.base_generator.base_generator.generate_task_sequence_config(
                env=env,
                dataset=dataset,
                task_sequence=[task, task],  # 同一个任务两次
                mode_sequence=["text_only", "multimodal"],  # 先text_only，再multimodal
                strategy="none",            # 无持续学习策略（只是顺序训练）
                use_label_embedding=False,
                seq_suffix=f"_{ablation_type}",
                **self.base_generator.recommended_hyperparams
            )
            
            # 添加CRF和Span Loss配置到每个session
            ablation_config = self.base_generator.ablation_configs[ablation_type]
            for task_config in config["tasks"]:
                # 只对序列任务启用CRF
                if task_config["task_name"] in ["mate", "mner", "mabsa"]:
                    task_config.update({
                        "use_crf": ablation_config["use_crf"],
                        "use_span_loss": ablation_config["use_span_loss"],
                        "boundary_weight": ablation_config["boundary_weight"],
                        "span_f1_weight": ablation_config["span_f1_weight"],
                        "transition_weight": ablation_config["transition_weight"]
                    })
                    
                    # 确保num_labels正确
                    if task_config["task_name"] == "mate":
                        task_config["num_labels"] = 3
                    elif task_config["task_name"] == "mner":
                        task_config["num_labels"] = 9
                    elif task_config["task_name"] == "mabsa":
                        task_config["num_labels"] = 7
            
            # 添加消融实验元信息
            config["ablation_info"] = {
                "purpose": "Ablation study for CRF and Span Loss",
                "ablation_type": ablation_type,
                "configuration": ablation_config["description"],
                "mode_sequence": ["text_only", "multimodal"],
                "expected_improvement": self.base_generator._get_expected_improvement(ablation_type)
            }
            
            # Kaggle特殊配置
            config["kaggle_mode"] = True
            config["kaggle_output_path"] = "/kaggle/working"
            
            # 保存配置
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            configs.append({
                "file": config_name,
                "path": config_file.as_posix(),
                "task": task,
                "ablation_type": ablation_type,
                "dataset": dataset
            })
            
            print(f"  ✓ Generated: {config_name}")
        
        # 生成账号索引文件
        index_file = output_path / f"{account_id}_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({
                "account_id": account_id,
                "account_name": assignment["name"],
                "task": task,
                "ablations": ablations,
                "description": assignment["description"],
                "estimated_time_minutes": len(ablations) * self.time_estimates[task],
                "total_configs": len(configs),
                "configs": configs
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n  ✓ Index file: {index_file}\n")
        
        return configs, output_path
    
    def generate_all_accounts(self,
                             env: str = "server",
                             dataset: str = "twitter2015",
                             output_dir: str = "scripts/configs/kaggle_ablation"):
        """为所有6个账号生成配置"""
        
        all_configs = {}
        
        for account_id in self.account_assignments.keys():
            configs, account_dir = self.generate_account_configs(
                account_id=account_id,
                env=env,
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
                "description": "6账号Ablation Study总索引",
                "total_accounts": len(self.account_assignments),
                "total_configs": sum(len(v["configs"]) for v in all_configs.values()),
                "accounts": {
                    acc_id: {
                        "name": self.account_assignments[acc_id]["name"],
                        "task": self.account_assignments[acc_id]["task"],
                        "ablations": self.account_assignments[acc_id]["ablations"],
                        "configs_count": len(all_configs[acc_id]["configs"]),
                        "directory": all_configs[acc_id]["directory"]
                    }
                    for acc_id in self.account_assignments.keys()
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"✅ 所有账号配置生成完成")
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
配置: {", ".join(assignment["ablations"])}
预计时间: {len(configs) * self.time_estimates[assignment["task"]]} 分钟

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
print("{assignment['name']}")
print("="*80)
print(f"任务: {assignment['task'].upper()}")
print(f"配置数: {len(configs)}")
print(f"预计时间: {len(configs) * self.time_estimates[assignment['task']]} 分钟")
print("="*80 + "\\n")

# ============================================================================
# Step 1: 项目设置
# ============================================================================

print("\\n" + "="*80)
print("Step 1: 设置项目")
print("="*80)

# 检查数据集
dataset_path = Path(KAGGLE_INPUT) / PROJECT_DATASET
if not dataset_path.exists():
    print(f"❌ 数据集未找到: {{dataset_path}}")
    print("请在Notebook设置中添加 '{{PROJECT_DATASET}}' 数据集")
    sys.exit(1)

print(f"✓ 数据集路径: {{dataset_path}}")

# 复制项目到工作目录
project_dir = Path(KAGGLE_WORKING) / "MCM"
if project_dir.exists():
    print(f"清理旧项目: {{project_dir}}")
    shutil.rmtree(project_dir)

print(f"复制项目到: {{project_dir}}")
shutil.copytree(dataset_path, project_dir)

# 切换到项目目录
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

print(f"✓ 当前工作目录: {{os.getcwd()}}")
print(f"✓ Python路径已更新")

# ============================================================================
# Step 2: 检查依赖
# ============================================================================

print("\\n" + "="*80)
print("Step 2: 检查依赖")
print("="*80)

try:
    import torch
    print(f"✓ PyTorch: {{torch.__version__}}")
    print(f"✓ CUDA available: {{torch.cuda.is_available()}}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {{torch.cuda.get_device_name(0)}}")
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

print("\\n" + "="*80)
print("Step 3: 运行实验")
print("="*80)

# 配置文件列表
configs = {json.dumps([{"file": c["file"], "ablation": c["ablation_type"]} for c in configs], indent=2)}

results = []
start_time = time.time()

for i, config_info in enumerate(configs, 1):
    config_file = Path("scripts/configs/kaggle_ablation/{account_id}") / config_info["file"]
    ablation_type = config_info["ablation"]
    
    print(f"\\n{{'-'*80}}")
    print(f"实验 {{i}}/{{len(configs)}}: {{ablation_type}}")
    print(f"配置: {{config_file}}")
    print(f"{{'-'*80}}")
    
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
            env={{**os.environ, "PYTHONPATH": str(project_dir)}},
            capture_output=True,
            text=True
        )
        
        exp_time = time.time() - exp_start
        
        if result.returncode == 0:
            print(f"✅ 实验 {{i}} 完成 ({{exp_time/60:.1f}} 分钟)")
            status = "success"
        else:
            print(f"❌ 实验 {{i}} 失败")
            print(f"错误: {{result.stderr[-500:]}}")
            status = "failed"
        
        results.append({{
            "experiment_id": i,
            "ablation_type": ablation_type,
            "status": status,
            "time_minutes": exp_time / 60,
            "config_file": str(config_file)
        }})
        
    except Exception as e:
        print(f"❌ 实验 {{i}} 异常: {{e}}")
        results.append({{
            "experiment_id": i,
            "ablation_type": ablation_type,
            "status": "error",
            "error": str(e)
        }})
    
    # 保存中间结果
    with open(Path(KAGGLE_WORKING) / "{account_id}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

# ============================================================================
# Step 4: 分析结果
# ============================================================================

print("\\n" + "="*80)
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
            print(f"⚠️ 无法读取 {{json_file}}: {{e}}")

print(f"✓ 找到 {{len(all_metrics)}} 个结果文件")

# 生成对比分析
if len(all_metrics) >= 2:
    print("\\n" + "-"*80)
    print("结果对比:")
    print("-"*80)
    
    # 按ablation类型分组
    baseline_results = [m for m in all_metrics if m["ablation_type"] == "baseline"]
    crf_results = [m for m in all_metrics if m["ablation_type"] == "crf_only"]
    
    if baseline_results and crf_results:
        # 取最新的结果（如果有多个）
        baseline = baseline_results[-1]
        crf = crf_results[-1]
        
        print(f"\\n{{assignment['task'].upper()}} 任务:")
        print(f"  Baseline:")
        print(f"    - Token Acc: {{baseline.get('token_accuracy', 'N/A'):.4f if isinstance(baseline.get('token_accuracy'), (int, float)) else 'N/A'}}")
        print(f"    - Chunk F1:  {{baseline.get('chunk_f1', 'N/A'):.4f if isinstance(baseline.get('chunk_f1'), (int, float)) else 'N/A'}}")
        
        print(f"  CRF Only:")
        print(f"    - Token Acc: {{crf.get('token_accuracy', 'N/A'):.4f if isinstance(crf.get('token_accuracy'), (int, float)) else 'N/A'}}")
        print(f"    - Chunk F1:  {{crf.get('chunk_f1', 'N/A'):.4f if isinstance(crf.get('chunk_f1'), (int, float)) else 'N/A'}}")
        
        # 计算提升
        if isinstance(baseline.get('chunk_f1'), (int, float)) and isinstance(crf.get('chunk_f1'), (int, float)):
            improvement = crf['chunk_f1'] - baseline['chunk_f1']
            improvement_pct = (improvement / baseline['chunk_f1']) * 100 if baseline['chunk_f1'] > 0 else 0
            print(f"  Improvement:")
            print(f"    - Chunk F1: +{{improvement:.4f}} ({{improvement_pct:+.1f}}%)")
else:
    print("⚠️ 结果不足，无法生成对比分析")

# ============================================================================
# Step 5: 保存结果
# ============================================================================

total_time = time.time() - start_time

print("\\n" + "="*80)
print("Step 5: 保存结果")
print("="*80)

# 保存执行摘要
final_results = {{
    "account_id": "{account_id}",
    "account_name": "{assignment['name']}",
    "task": "{assignment['task']}",
    "ablations": {assignment['ablations']},
    "total_experiments": len(results),
    "successful": sum(1 for r in results if r['status'] == 'success'),
    "failed": sum(1 for r in results if r['status'] in ['failed', 'error']),
    "total_time_minutes": total_time / 60,
    "experiments": results,
    "metrics": all_metrics
}}

output_file = Path(KAGGLE_WORKING) / "{account_id}_final_results.json"
with open(output_file, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"✓ 结果已保存: {{output_file}}")

# 打包所有结果
print("\\n打包结果文件...")
import zipfile

# 打包1: metrics和摘要JSON
zip_path = Path(KAGGLE_WORKING) / "{account_id}_results.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    # 添加摘要JSON
    zipf.write(output_file, arcname=output_file.name)
    
    # 添加所有metrics JSON
    for metrics in all_metrics:
        file_path = Path(metrics["file_path"])
        if file_path.exists():
            zipf.write(file_path, arcname=f"metrics/{{file_path.name}}")

print(f"✓ 结果已打包: {{zip_path}}")

# 打包2: 单独压缩所有模型文件（.pt）
print("\\n压缩模型文件...")
models_zip_path = Path(KAGGLE_WORKING) / "{account_id}_models.zip"
pt_files_count = 0

with zipfile.ZipFile(models_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    # 遍历工作目录查找所有 .pt 文件
    for pt_file in Path(KAGGLE_WORKING).rglob("*.pt"):
        # 添加到压缩包，保留相对路径结构
        arcname = pt_file.relative_to(KAGGLE_WORKING)
        zipf.write(pt_file, arcname=str(arcname))
        pt_files_count += 1
        print(f"  压缩: {{arcname}}")

print(f"✓ 模型文件已压缩: {{models_zip_path}}")
print(f"  共 {{pt_files_count}} 个模型文件")

print("\\n" + "="*80)
print("✅ 所有任务完成！")
print("="*80)
print(f"总实验数: {{len(results)}}")
print(f"成功: {{sum(1 for r in results if r['status'] == 'success')}}")
print(f"失败: {{sum(1 for r in results if r['status'] in ['failed', 'error'])}}")
print(f"总时间: {{total_time/60:.1f}} 分钟")
print("\\n请下载以下文件:")
print(f"  1. {{output_file.name}} - JSON格式的详细结果")
print(f"  2. {{zip_path.name}} - 打包的指标和结果文件")
print(f"  3. {{models_zip_path.name}} - 打包的所有模型文件（.pt）")
print("="*80)
''')
        
        print(f"  ✓ Runner script: {runner_path}")
    
    def _generate_deployment_guide(self, output_dir: Path):
        """生成部署指南"""
        
        guide_path = output_dir / "KAGGLE_DEPLOYMENT_GUIDE.md"
        
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write("""# Kaggle 6账号Ablation Study部署指南

## 📋 实验设计

### 总体方案

- **总配置数**: 12个 (3任务 × 4 ablation)
- **账号数**: 6个
- **每账号配置数**: 2个
- **每账号预计时间**: 3-4小时
- **Kaggle限制**: 12小时（充足余量）

### 账号分配

| 账号 | 任务 | Ablation配置 | 预计时间 |
|------|------|--------------|----------|
| **Account 1** | MATE | baseline + crf_and_span | ~3小时 |
| **Account 2** | MATE | crf_only + span_only | ~3小时 |
| **Account 3** | MNER | baseline + crf_and_span | ~3.3小时 |
| **Account 4** | MNER | crf_only + span_only | ~3.3小时 |
| **Account 5** | MABSA | baseline + crf_and_span | ~3.6小时 |
| **Account 6** | MABSA | crf_only + span_only | ~3.6小时 |

## 🚀 部署步骤

### 1. 准备Kaggle数据集

在本地执行以下步骤：

```bash
# 1. 打包项目
cd /path/to/MCM
zip -r mcm-project.zip . -x "*.git*" "*__pycache__*" "*.pyc" "*checkpoint*" "*logs*"

# 2. 上传到Kaggle
# 在Kaggle网站上:
# - 点击 "Datasets" -> "New Dataset"
# - 上传 mcm-project.zip
# - 设置名称为 "mcm-project"
# - 设置为 Private
# - 点击 "Create"
```

### 2. 为每个账号创建Notebook

对于每个账号（1-6），重复以下步骤：

#### Step 1: 创建Notebook

1. 登录对应的Kaggle账号
2. 点击 "Code" -> "New Notebook"
3. Notebook设置：
   - **加速器**: GPU P100
   - **Internet**: On
   - **持久化**: Off (节省配额)

#### Step 2: 添加数据集

1. 点击右侧 "Add Data"
2. 搜索 "mcm-project"
3. 添加你的数据集

#### Step 3: 复制运行脚本

1. 打开对应账号的运行脚本:
   - Account 1: `account_1/run_account_1.py`
   - Account 2: `account_2/run_account_2.py`
   - ... (以此类推)

2. 复制全部内容到Notebook

3. 修改数据集名称（如果需要）:
   ```python
   PROJECT_DATASET = "your-username/mcm-project"  # 修改为你的数据集路径
   ```

#### Step 4: 运行

1. 点击 "Save Version"
2. 选择 "Save & Run All (Commit)"
3. 等待完成（3-4小时）

### 3. 同时运行所有账号

✨ **关键**: 6个账号可以同时运行，互不干扰！

```
Account 1  →  [MATE baseline + full]     →  3小时  →  完成
Account 2  →  [MATE crf + span]          →  3小时  →  完成
Account 3  →  [MNER baseline + full]     →  3.3小时 →  完成
Account 4  →  [MNER crf + span]          →  3.3小时 →  完成
Account 5  →  [MABSA baseline + full]    →  3.6小时 →  完成
Account 6  →  [MABSA crf + span]         →  3.6小时 →  完成

总时间: ~3.6小时 (并行)
```

## 📊 收集结果

### 每个账号完成后

1. 进入对应的Notebook
2. 点击 "Output"
3. 下载文件:
   - `account_X_final_results.json` (必需)
   - `checkpoints/` 目录下的模型文件 (可选)

### 文件组织

```
results/
├── account_1_final_results.json
├── account_2_final_results.json
├── account_3_final_results.json
├── account_4_final_results.json
├── account_5_final_results.json
└── account_6_final_results.json
```

## 🔍 结果分析

下载所有结果后，运行分析脚本：

```bash
cd scripts/configs/kaggle_ablation

# 将下载的JSON文件放到results/目录

python analyze_results.py
```

这将生成：
- `ablation_study_summary.json` - 总结
- `ablation_study_report.md` - 详细报告
- `ablation_comparison.png` - 对比图表

## ⚠️ 注意事项

### Kaggle限制

- **GPU时间**: 每周30小时（每账号）
- **运行时间**: 单次最多12小时
- **并行**: 每账号最多1个active session

### 最佳实践

1. **监控进度**: 定期检查Output日志
2. **及时保存**: 实验完成后立即下载结果
3. **备份**: 保存所有JSON文件
4. **网络稳定**: 确保上传数据集时网络稳定

### 故障处理

**问题1: 数据集未找到**
```
❌ 数据集未找到: /kaggle/input/mcm-project
```
解决: 检查数据集是否正确添加，名称是否匹配

**问题2: GPU不可用**
```
❌ CUDA available: False
```
解决: 检查Notebook设置，确保选择了GPU P100加速器

**问题3: 时间超限**
```
Session timeout after 12 hours
```
解决: 减少epochs或batch_size（但我们的配置应该不会超时）

## 📈 预期结果

### Chunk F1提升

| 配置 | MATE | MNER | MABSA |
|------|------|------|-------|
| **Baseline** | ~32% | ~30% | ~35% |
| **CRF only** | ~68% (+36%) | ~65% (+35%) | ~70% (+35%) |
| **Span only** | ~65% (+33%) | ~62% (+32%) | ~67% (+32%) |
| **CRF + Span** | ~76% (+44%) | ~74% (+44%) | ~78% (+43%) |

## 🎉 完成检查清单

- [ ] 6个账号的Notebook都已创建
- [ ] 所有Notebook都添加了mcm-project数据集
- [ ] 所有Notebook都设置了GPU P100
- [ ] 所有运行脚本已正确复制
- [ ] 6个Notebook同时运行
- [ ] 所有结果文件已下载
- [ ] 结果分析脚本已运行
- [ ] 分析报告已生成

---

**准备好了吗？开始你的Ablation Study之旅！** 🚀
""")
        
        print(f"  ✓ Deployment guide: {guide_path}")
    
    def _generate_analysis_script(self, output_dir: Path, all_configs: dict):
        """生成结果分析脚本"""
        
        script_path = output_dir / "analyze_results.py"
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('''#!/usr/bin/env python3
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
        f.write("# Ablation Study Results\\n\\n")
        f.write(f"## Summary\\n\\n")
        f.write(f"- Total Experiments: {summary['total_experiments']}\\n")
        f.write(f"- Successful: {summary['successful_experiments']}\\n")
        f.write(f"- Total Time: {summary['total_time_hours']:.1f} hours\\n")
        f.write(f"\\n## Results by Task\\n\\n")
        
        for task in summary['tasks']:
            task_df = df[df['task'] == task]
            f.write(f"### {task.upper()}\\n\\n")
            f.write(task_df.to_markdown(index=False))
            f.write("\\n\\n")
    
    print(f"✓ Report saved: {output_path / 'ablation_study_report.md'}")

def main():
    print("="*80)
    print("Ablation Study结果分析")
    print("="*80)
    
    # 加载结果
    all_results = load_all_results()
    
    if not all_results:
        print("\\n❌ 没有找到结果文件")
        print("请确保将所有 account_X_final_results.json 放在 results/ 目录")
        return
    
    print(f"\\n✓ 加载了 {len(all_results)} 个账号的结果\\n")
    
    # 提取指标
    df = extract_metrics(all_results)
    
    # 生成报告
    generate_report(df)
    
    print("\\n✅ 分析完成！")

if __name__ == "__main__":
    main()
''')
        
        print(f"  ✓ Analysis script: {script_path}")


def main():
    parser = argparse.ArgumentParser(
        description="生成Kaggle 6账号Ablation Study配置"
    )
    parser.add_argument("--env", type=str, default="server",
                       choices=["local", "server"],
                       help="环境类型")
    parser.add_argument("--dataset", type=str, default="twitter2015",
                       choices=["twitter2015", "twitter2017"],
                       help="数据集名称")
    parser.add_argument("--output_dir", type=str,
                       default="scripts/configs/kaggle_ablation",
                       help="输出目录")
    parser.add_argument("--account", type=str, default="all",
                       help="指定账号ID (account_1 到 account_6) 或 'all'")
    
    args = parser.parse_args()
    
    generator = KaggleAblationStudyGenerator()
    
    if args.account == "all":
        # 为所有账号生成配置
        generator.generate_all_accounts(
            env=args.env,
            dataset=args.dataset,
            output_dir=args.output_dir
        )
    else:
        # 只为指定账号生成配置
        generator.generate_account_configs(
            account_id=args.account,
            env=args.env,
            dataset=args.dataset,
            output_dir=args.output_dir
        )
    
    print("\n" + "="*80)
    print("下一步:")
    print("="*80)
    print("1. 查看部署指南: scripts/configs/kaggle_ablation/KAGGLE_DEPLOYMENT_GUIDE.md")
    print("2. 准备Kaggle数据集")
    print("3. 在6个账号上同时运行")
    print("4. 下载结果JSON文件")
    print("5. 运行结果分析: python scripts/configs/kaggle_ablation/analyze_results.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

