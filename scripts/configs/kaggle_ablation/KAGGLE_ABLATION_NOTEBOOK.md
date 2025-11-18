# CRF & Span Loss 消融实验 - Kaggle Notebook

本文档提供了在 Kaggle 上运行消融实验的完整 Notebook 代码。

## 📋 实验概述

**目标**: 对比 4 种配置在 MATE、MNER、MABSA 任务上的效果
- **Baseline**: 无 CRF 和 Span Loss
- **CRF Only**: 仅启用 CRF
- **Span Only**: 仅启用 Span Loss
- **Both**: 同时启用 CRF 和 Span Loss

**账号分配**:
- Account 1: MATE (Baseline + Both)
- Account 2: MATE (CRF Only + Span Only)
- Account 3: MNER (Baseline + Both)
- Account 4: MNER (CRF Only + Span Only)
- Account 5: MABSA (Baseline + Both)
- Account 6: MABSA (CRF Only + Span Only)

## 🚀 使用方法

1. 在 Kaggle 创建新 Notebook
2. 复制以下所有 Cell 代码
3. **仅修改 Cell 1 中的账号编号**
4. 添加数据集（mcm-code 和 mcm-data）
5. 启用 GPU (P100 或 T4)
6. 运行 "Run All"

---

## 📝 Notebook Cells

### Cell 1: 配置账号编号 ⚙️ （**唯一需要修改的地方**）

```python
"""
⚠️⚠️⚠️ 重要：这是唯一需要修改的地方！⚠️⚠️⚠️

在不同的 Kaggle 账号上运行时，修改下面的 ACCOUNT_ID：
- Account 1 → ACCOUNT_ID = 1
- Account 2 → ACCOUNT_ID = 2
- Account 3 → ACCOUNT_ID = 3
- Account 4 → ACCOUNT_ID = 4
- Account 5 → ACCOUNT_ID = 5
- Account 6 → ACCOUNT_ID = 6
"""

# ============ 修改这里！ ============
ACCOUNT_ID = 1  # 👈 根据当前账号修改为 1-6
# ==================================

print("="*80)
print("CRF & Span Loss 消融实验")
print("="*80)
print(f"\n✅ 账号配置: Account {ACCOUNT_ID}")
print(f"📂 配置目录: account_{ACCOUNT_ID}/")

# 账号任务映射
ACCOUNT_TASKS = {
    1: ("MATE", ["baseline", "crf_and_span"]),
    2: ("MATE", ["crf_only", "span_only"]),
    3: ("MNER", ["baseline", "crf_and_span"]),
    4: ("MNER", ["crf_only", "span_only"]),
    5: ("MABSA", ["baseline", "crf_and_span"]),
    6: ("MABSA", ["crf_only", "span_only"]),
}

task_name, configs = ACCOUNT_TASKS[ACCOUNT_ID]
print(f"📋 任务: {task_name}")
print(f"🧪 实验配置: {', '.join(configs)}")
print(f"📊 实验数量: {len(configs)}")

print("\n✅ 配置完成")
```

---

### Cell 2: 环境检查

```python
import os
import sys
import shutil
from pathlib import Path

print("="*80)
print("环境检查")
print("="*80)

# 检查Kaggle环境
print("\n📦 可用数据集:")
for dataset in os.listdir("/kaggle/input"):
    print(f"  - {dataset}")

# 自动检测模式
use_split_mode = False
code_path = None
data_path = None

# 检测分离模式
if os.path.exists("/kaggle/input/mcm-code"):
    use_split_mode = True
    code_path = Path("/kaggle/input/mcm-code")
    print("\n✓ 检测到分离模式")
    print(f"  代码路径: {code_path}")
  
    if os.path.exists("/kaggle/input/mcm-data"):
        data_path = Path("/kaggle/input/mcm-data")
        print(f"  数据路径: {data_path}")
    else:
        print("  ⚠️ 未找到 mcm-data，请在Data面板添加")

# 检测完整模式
else:
    possible_paths = [
        Path("/kaggle/input/mcm-project/MCM"),
        Path("/kaggle/input/mcm-project"),
    ]
  
    for path in possible_paths:
        if path.exists() and (path / "scripts").exists():
            code_path = path
            print(f"\n✓ 检测到完整模式")
            print(f"  项目路径: {path}")
            break

if code_path is None:
    raise FileNotFoundError("❌ 未找到项目！请检查数据集配置")

# 列出项目内容
print("\n📁 项目内容（前10项）:")
items = sorted(list(code_path.iterdir()))[:10]
for item in items:
    print(f"  - {item.name}{'/' if item.is_dir() else ''}")

print("\n✅ 环境检查完成")
```

---

### Cell 3: 复制项目到工作目录

```python
# 复制项目到可写目录
work_project_path = Path("/MCM")

print("="*80)
print("复制项目到工作目录")
print("="*80)

if not work_project_path.exists():
    print(f"\n📋 复制代码...")
    print(f"  源: {code_path}")
    print(f"  目标: {work_project_path}")
    shutil.copytree(code_path, work_project_path, dirs_exist_ok=True)
    print("✓ 代码复制完成")
else:
    print("⚠️ 工作目录已存在，跳过复制")

# 如果是分离模式，链接数据目录
if use_split_mode and data_path:
    print("\n📋 链接数据目录（分离模式）...")
  
    target_data = work_project_path / "data"
    target_model = work_project_path / "downloaded_model"
  
    # 链接data
    if not target_data.exists():
        source_data = data_path / "data" if (data_path / "data").exists() else data_path
        print(f"  data: {source_data} → {target_data}")
        try:
            os.symlink(source_data, target_data)
            print("  ✓ data链接成功（符号链接）")
        except:
            print("  ⚠️ 符号链接失败，改用复制...")
            shutil.copytree(source_data, target_data, dirs_exist_ok=True)
            print("  ✓ data复制完成")
    else:
        print(f"  ✓ data目录已存在")
  
    # 链接模型
    source_model = data_path / "downloaded_model"
    if source_model.exists() and not target_model.exists():
        print(f"  downloaded_model: {source_model} → {target_model}")
        try:
            os.symlink(source_model, target_model)
            print("  ✓ downloaded_model链接成功（符号链接）")
        except:
            shutil.copytree(source_model, target_model, dirs_exist_ok=True)
            print("  ✓ downloaded_model复制完成")
    else:
        print(f"  ✓ downloaded_model目录已存在")

# 切换工作目录
os.chdir(work_project_path)
sys.path.insert(0, str(work_project_path))

print(f"\n📂 当前工作目录: {os.getcwd()}")
print(f"🐍 Python路径: {sys.path[0]}")

# 验证数据集
data_dir = work_project_path / "data"
print(f"\n📁 数据目录: {data_dir}")
print(f"   存在: {data_dir.exists()}")

if data_dir.exists():
    print("\n📦 可用数据集:")
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            file_count = len(list(item.iterdir()))
            print(f"  - {item.name}/ ({file_count} files)")

# 验证配置目录
config_dir = work_project_path / "scripts/configs/kaggle_ablation" / f"account_{ACCOUNT_ID}"
print(f"\n📂 账号配置目录: {config_dir}")
print(f"   存在: {config_dir.exists()}")

if config_dir.exists():
    print("\n📄 配置文件:")
    for item in sorted(config_dir.iterdir()):
        if item.suffix == ".json":
            print(f"  - {item.name}")

print("\n✅ 项目准备完成")
```

---

### Cell 4: 安装依赖

```python
import subprocess

print("="*80)
print("安装依赖")
print("="*80)

# 检查依赖文件
kaggle_req = work_project_path / "requirements_kaggle.txt"
regular_req = work_project_path / "requirements.txt"

print("\n📦 检查依赖文件...")

if kaggle_req.exists():
    print("✓ 找到 requirements_kaggle.txt（Kaggle优化版）")
    print("\n安装Kaggle特定依赖...")
    !pip install -q -r {str(kaggle_req)}
    print("✓ 依赖安装完成")
  
elif regular_req.exists():
    print("✓ 找到 requirements.txt（标准版）")
    print("\n⚠️ 可能有版本冲突警告（可以忽略）")
    !pip install -q -r {str(regular_req)}
    print("\n✓ 依赖安装完成（版本冲突警告可忽略）")
  
else:
    print("⚠️ 未找到依赖文件，安装最小依赖集...")
    !pip install -q pytorch_crf sentencepiece protobuf==3.20.3
    print("✓ 最小依赖安装完成")

# 验证关键包
print("\n🔍 验证关键依赖...")
try:
    import torch
    print(f"  ✓ torch: {torch.__version__}")
except:
    print("  ✗ torch导入失败")

try:
    import transformers
    print(f"  ✓ transformers: {transformers.__version__}")
except:
    print("  ✗ transformers导入失败")

try:
    from torchcrf import CRF
    print(f"  ✓ torchcrf: 可用")
except:
    print("  ⚠️ torchcrf不可用（将使用内置SimpleCRF）")

print("\n💡 说明:")
print("  • Kaggle已预装大部分包")
print("  • 版本冲突警告通常可以忽略")
print("  • torchcrf不可用时会自动使用SimpleCRF")

print("\n✅ 依赖检查完成")
```

---

### Cell 5: 检查GPU和验证修复

```python
import torch

print("="*80)
print("GPU信息 & 修复验证")
print("="*80)

# GPU信息
print("\n🖥️ GPU状态:")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
  
    print(f"  ✓ GPU: {gpu_name}")
    print(f"  ✓ 总显存: {gpu_memory:.1f} GB")
    print(f"  ✓ CUDA: {torch.version.cuda}")
    print(f"  ✓ PyTorch: {torch.__version__}")
else:
    print("  ❌ 未检测到GPU")
    print("  请在Settings → Accelerator中选择GPU")

# 验证CRF和Span Loss修复
print("\n🔧 验证修复是否已集成:")

# 1. 检查CRF是否可用
try:
    from models.task_heads.mate_head import MATEHead
    head = MATEHead(768, 3, use_crf=True)
    if hasattr(head, 'crf') and head.crf is not None:
        print("  ✓ CRF层: 已集成")
        print(f"    类型: {type(head.crf).__name__}")
    else:
        print("  ✗ CRF层: 未找到")
except Exception as e:
    print(f"  ✗ CRF层检查失败: {e}")

# 2. 检查Span Loss
try:
    from utils.span_loss import SpanLoss
    span_loss = SpanLoss('mate')
    print("  ✓ Span Loss: 已集成")
except Exception as e:
    print(f"  ✗ Span Loss检查失败: {e}")

# 3. 检查训练循环
try:
    with open(work_project_path / "modules/training_loop_fixed.py", "r") as f:
        content = f.read()
        if "span_loss" in content.lower():
            print("  ✓ 训练循环: 已集成Span Loss")
        else:
            print("  ⚠️ 训练循环: 未找到Span Loss")
except:
    print("  ⚠️ 无法检查训练循环")

print("\n✅ 系统准备完成")
```

---

### Cell 6: 显示实验配置

```python
import json

print("="*80)
print(f"Account {ACCOUNT_ID} 消融实验配置")
print("="*80)

# 加载索引文件
index_file = work_project_path / "scripts/configs/kaggle_ablation" / f"account_{ACCOUNT_ID}" / f"account_{ACCOUNT_ID}_index.json"

if index_file.exists():
    with open(index_file) as f:
        index_data = json.load(f)
    
    # 从第一个配置中获取数据集名称
    dataset_name = index_data['configs'][0]['dataset'] if index_data['configs'] else 'twitter2015'
    
    print(f"\n📋 实验信息:")
    print(f"  账号: Account {index_data['account_id']}")
    print(f"  任务: {index_data['task'].upper()}")
    print(f"  数据集: {dataset_name}")
    print(f"  配置数量: {index_data['total_configs']}")
    
    print(f"\n🧪 实验配置:")
    for i, cfg in enumerate(index_data['configs'], 1):
        print(f"\n  配置 {i}/{index_data['total_configs']}:")
        print(f"    类型: {cfg['ablation_type']}")
        print(f"    文件: {cfg['file']}")
        
        # 读取配置详情
        cfg_file = work_project_path / "scripts/configs/kaggle_ablation" / f"account_{ACCOUNT_ID}" / cfg['file']
        if cfg_file.exists():
            with open(cfg_file) as f:
                cfg_data = json.load(f)
            
            # 显示session数量和模式序列
            total_tasks = cfg_data.get('total_tasks', 1)
            mode_seq = cfg_data.get('mode_sequence', ['multimodal'])
            print(f"    Sessions: {total_tasks}")
            print(f"    模式序列: {' → '.join(mode_seq)}")
            
            # 显示CRF和Span Loss配置（从第一个task获取）
            if cfg_data.get('tasks') and len(cfg_data['tasks']) > 0:
                first_task = cfg_data['tasks'][0]
                print(f"    CRF: {'✅' if first_task.get('use_crf', 0) else '❌'}")
                print(f"    Span Loss: {'✅' if first_task.get('use_span_loss', 0) else '❌'}")
    
    # 时间估算 - 每个配置包含2个sessions (text_only + multimodal)
    time_per_config = 3.5  # 小时（每个配置，包含2个sessions）
    total_time = index_data['total_configs'] * time_per_config
    
    print(f"\n⏱️ 预计时间:")
    print(f"  每个配置: ~{time_per_config} 小时 (text_only + multimodal)")
    print(f"  总计: ~{total_time} 小时")
    print(f"  Kaggle限制: 12 小时")
    
    if total_time > 12:
        print(f"  ⚠️ 预计时间可能超过Kaggle限制")
    else:
        print(f"  ✅ 预计在时间限制内")
    
else:
    print(f"❌ 索引文件不存在: {index_file}")
    raise FileNotFoundError(f"配置索引文件不存在")

print("\n✅ 配置检查完成")
```

---

### Cell 7: 运行消融实验 🚀

```python
import json
import subprocess
import time
from datetime import datetime

print("="*80)
print(f"开始运行 Account {ACCOUNT_ID} 消融实验")
print("="*80)
print()

# 确保输出目录存在
output_dir = Path("/kaggle/working/checkpoints")
output_dir.mkdir(parents=True, exist_ok=True)

# 加载实验索引
index_file = work_project_path / "scripts/configs/kaggle_ablation" / f"account_{ACCOUNT_ID}" / f"account_{ACCOUNT_ID}_index.json"

with open(index_file) as f:
    index_data = json.load(f)

total_configs = index_data['total_configs']
task_name = index_data['task'].upper()  # 所有配置都是同一个任务
print(f"📊 总配置数: {total_configs}")
print(f"📋 任务: {task_name}")
print(f"💡 说明: 每个配置包含 text_only 和 multimodal 两个session")
print()

# 运行每个实验
results = []
overall_start = time.time()

for i, cfg_info in enumerate(index_data['configs'], 1):
    config_name = cfg_info['ablation_type']  # baseline, crf_only, span_only, crf_and_span
    config_file = work_project_path / "scripts/configs/kaggle_ablation" / f"account_{ACCOUNT_ID}" / cfg_info['file']
    
    print("="*80)
    print(f"实验 [{i}/{total_configs}]: {task_name} - {config_name}")
    print("="*80)
    print(f"配置文件: {cfg_info['file']}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 更新配置文件路径到Kaggle输出目录
    print("📝 更新配置文件路径...")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # 更新输出路径
    original_output = config["global_params"]["output_model_path"]
    original_train_info = config["global_params"]["train_info_json"]
    
    config["global_params"]["output_model_path"] = f"/kaggle/working/checkpoints/{Path(original_output).name}"
    config["global_params"]["train_info_json"] = f"/kaggle/working/checkpoints/{Path(original_train_info).name}"
    
    # 更新其他路径
    if "ewc_dir" in config["global_params"]:
        config["global_params"]["ewc_dir"] = "/kaggle/working/checkpoints/ewc"
    if "gem_mem_dir" in config["global_params"]:
        config["global_params"]["gem_mem_dir"] = "/kaggle/working/checkpoints/gem"
    
    # 保存更新后的配置
    temp_config = Path("/kaggle/working") / f"temp_config_{config_name}.json"
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  ✓ 输出路径更新到: /kaggle/working/checkpoints/")
    print()
    
    # 运行实验
    exp_start = time.time()
    
    # 读取配置获取实际任务数
    with open(temp_config, 'r') as f:
        temp_cfg = json.load(f)
    total_tasks_in_config = temp_cfg.get('total_tasks', 2)
    
    cmd = [
        "python", "-m", "scripts.train_with_zero_shot",
        "--config", str(temp_config),
        "--start_task", "0",
        "--end_task", str(total_tasks_in_config)  # 使用实际任务数
    ]
    
    # 设置环境变量
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(work_project_path)
    
    try:
        print(f"🚀 运行命令: {' '.join(cmd)}")
        print(f"   工作目录: {work_project_path}")
        print(f"   PYTHONPATH: {env['PYTHONPATH']}")
        print()
        
        result = subprocess.run(cmd, check=True, capture_output=False, 
                               cwd=str(work_project_path), env=env)
        success = True
        print()
        print(f"✅ {config_name} 实验完成")
    except subprocess.CalledProcessError as e:
        success = False
        print()
        print(f"❌ {config_name} 实验失败: {e}")
    
    exp_time = time.time() - exp_start
    
    # 验证输出文件
    output_files = list(output_dir.glob("**/*"))
    output_files = [f for f in output_files if f.is_file()]
    
    print(f"⏱️ 耗时: {exp_time/60:.1f} 分钟")
    print(f"📁 当前输出文件数: {len(output_files)}")
    
    # 记录结果
    results.append({
        "config": config_name,
        "file": cfg_info['file'],
        "success": success,
        "time_minutes": round(exp_time/60, 1),
        "output_files": len(output_files)
    })
    
    # 保存进度
    progress_file = Path("/kaggle/working/ablation_progress.json")
    with open(progress_file, 'w') as f:
        json.dump({
            "account_id": ACCOUNT_ID,
            "completed": i,
            "total": total_configs,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print()

# 总结
total_time = time.time() - overall_start
success_count = sum(1 for r in results if r['success'])

print("="*80)
print("🎉 所有实验完成!")
print("="*80)
print(f"\n📊 实验统计:")
print(f"  账号: Account {ACCOUNT_ID}")
print(f"  任务: {task_name}")
print(f"  总实验数: {total_configs}")
print(f"  成功: {success_count}")
print(f"  失败: {total_configs - success_count}")
print(f"  总耗时: {total_time/3600:.2f} 小时")

print(f"\n📋 详细结果:")
for r in results:
    status = "✅" if r['success'] else "❌"
    print(f"  {status} {r['config']}: {r['time_minutes']:.1f} 分钟")

print(f"\n💾 结果保存在: /kaggle/working/checkpoints/")
print(f"📝 进度文件: /kaggle/working/ablation_progress.json")

print("\n✅ 实验完成")
```

---

### Cell 8: 打包结果（不进行分析）

```python
import shutil
from pathlib import Path
from datetime import datetime

print("="*80)
print("打包实验结果")
print("="*80)

output_dir = Path("/kaggle/working/checkpoints")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_zip = Path(f"/kaggle/working/account_{ACCOUNT_ID}_results_{timestamp}.zip")

if output_dir.exists():
    # 统计文件
    all_files = list(output_dir.rglob("*"))
    files = [f for f in all_files if f.is_file()]
    train_info_files = [f for f in files if "train_info" in f.name and f.suffix == ".json"]
    model_files = [f for f in files if f.suffix == ".pt"]
    
    print(f"\n📁 输出目录: {output_dir}")
    print(f"  总文件数: {len(files)}")
    print(f"  train_info: {len(train_info_files)}")
    print(f"  模型文件: {len(model_files)}")
    
    # 计算总大小
    total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
    print(f"  总大小: {total_size:.1f} MB")
    
    print("\n📦 正在打包...")
    print(f"  源目录: {output_dir}")
    print(f"  目标文件: {output_zip.name}")
    
    # 创建压缩包
    shutil.make_archive(
        str(output_zip.with_suffix('')),
        'zip',
        output_dir
    )
    
    if output_zip.exists():
        zip_size = output_zip.stat().st_size / (1024 * 1024)
        print(f"\n✅ 结果已打包!")
        print(f"  文件: {output_zip.name}")
        print(f"  大小: {zip_size:.1f} MB")
        
        print(f"\n📥 下载方式:")
        print(f"  1. 点击右侧 'Output' 标签")
        print(f"  2. 找到 {output_zip.name}")
        print(f"  3. 点击下载按钮")
        
        print(f"\n💡 文件命名说明:")
        print(f"  account_{ACCOUNT_ID}_results_{timestamp}.zip")
        print(f"  └─ 账号{ACCOUNT_ID}的实验结果")
        print(f"     └─ 时间戳: {timestamp}")
    else:
        print("❌ 打包失败")
else:
    print("❌ 输出目录不存在，无内容可打包")

print("\n" + "="*80)
print("📌 后续步骤:")
print("="*80)
print(f"  1. ✅ 下载 account_{ACCOUNT_ID}_results_{timestamp}.zip")
print(f"  2. ⏸️ 点击右上角 'Stop Session' 停止Notebook（节省GPU配额）")
print(f"  3. 🔄 重复以上步骤在其他账号上运行")
print(f"  4. 📊 所有账号完成后，使用本地的 analyze_results.py 分析")
print("="*80)

print("\n💡 分析说明:")
print("  • 本次运行不进行性能分析")
print("  • 等所有6个账号都完成后")
print("  • 将所有结果zip文件下载到本地")
print("  • 运行 scripts/configs/kaggle_ablation/analyze_results.py 进行综合分析")

print("\n✅ 打包完成")
```

---

## 📊 后续分析步骤

当所有 6 个账号都完成后：

### 1. 下载所有结果

将 6 个账号的结果 zip 文件下载到本地：
```
results/
  ├── account_1_results_YYYYMMDD_HHMMSS.zip
  ├── account_2_results_YYYYMMDD_HHMMSS.zip
  ├── account_3_results_YYYYMMDD_HHMMSS.zip
  ├── account_4_results_YYYYMMDD_HHMMSS.zip
  ├── account_5_results_YYYYMMDD_HHMMSS.zip
  └── account_6_results_YYYYMMDD_HHMMSS.zip
```

### 2. 解压所有文件

```bash
cd /path/to/MCM/results
for zip in account_*.zip; do
    unzip -q "$zip" -d "${zip%.zip}"
done
```

### 3. 运行综合分析

```bash
cd /path/to/MCM
python scripts/configs/kaggle_ablation/analyze_results.py --results_dir ./results
```

这将生成：
- 📊 对比表格（各任务在4种配置下的性能）
- 📈 可视化图表（Chunk F1, Token F1 等）
- 📝 详细分析报告

---

## ⚠️ 注意事项

### 每个账号的修改

1. **仅修改 Cell 1 中的 `ACCOUNT_ID`**
2. **其他 Cell 完全不需要修改**

### 时间管理

- 每个实验约 2.5 小时
- 每个账号 2 个实验 = 约 5 小时
- 远低于 Kaggle 12 小时限制 ✅

### GPU 配额

- 使用 P100 或 T4
- 实验完成后立即停止 Session
- 节省 GPU 配额

### 数据集

确保添加以下数据集到 Notebook：
- `mcm-code` (代码)
- `mcm-data` (数据和模型)

---

## 📋 检查清单

### 运行前
- [ ] 在 6 个账号上分别创建 Notebook
- [ ] 每个 Notebook 修改正确的 `ACCOUNT_ID`
- [ ] 添加 `mcm-code` 和 `mcm-data` 数据集
- [ ] 启用 GPU (P100 或 T4)
- [ ] 启用 Internet

### 运行中
- [ ] Cell 1: 账号配置正确
- [ ] Cell 2: 环境检查通过
- [ ] Cell 3: 项目复制成功
- [ ] Cell 4: 依赖安装完成
- [ ] Cell 5: GPU 和修复验证通过
- [ ] Cell 6: 配置显示正确
- [ ] Cell 7: 实验运行完成
- [ ] Cell 8: 结果打包成功

### 运行后
- [ ] 下载对应账号的结果 zip
- [ ] 停止 Session
- [ ] 所有 6 个账号完成后进行综合分析

---

Good luck with your ablation study! 🚀

