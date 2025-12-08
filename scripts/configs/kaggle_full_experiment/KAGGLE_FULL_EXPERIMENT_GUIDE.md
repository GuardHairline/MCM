# Kaggle 全量实验部署指南

本指南用于在 Kaggle 上运行 MCM 项目的 60 组全量对比实验。

## 目录结构准备

在本地运行 `scripts/generate_kaggle_full_configs.py` 后，你会得到 `scripts/configs/kaggle_full_experiment/` 文件夹，其中包含 60 个 JSON 文件。

### 1. 创建 Config 数据集
你需要将这些生成的 JSON 文件上传到 Kaggle 作为一个新的数据集。
1. 在 Kaggle 点击 "Create New Dataset"。
2. 拖入 `scripts/configs/kaggle_full_experiment/` 文件夹中的所有内容（包括 JSON 和 index）。
3. 命名为 `mcm-full-configs`。

### 2. 准备 Notebook 环境

创建一个新的 Kaggle Notebook，并添加以下**三个**数据集：
1. **mcm-code**: 包含代码（modules, scripts 等）。
2. **mcm-data**: 包含 `data`, `downloaded_model`, `reference` 文件夹。
3. **mcm-full-configs**: 包含刚才生成的 60 个 JSON 配置文件。

设置 Accelerator 为 **GPU P100**。

## 运行脚本模板

将以下代码复制到 Notebook 的第一个单元格中。这段代码会自动设置环境、读取配置并运行指定的实验。

**注意修改 `EXP_ID_START` 和 `EXP_ID_END` 来控制本次运行的任务。**

```python
# ==========================================
# MCM Kaggle Full Experiment Runner
# ==========================================

# >>> 设置本次运行的实验 ID 范围 (0-59) <<<
# 建议一次运行 1-2 个实验 (每个实验约 4-6 小时)
EXP_ID_START = 0  
EXP_ID_END = 1    # 运行范围 [START, END)

# ==========================================

import os
import sys
import json
import shutil
import subprocess
import time
from pathlib import Path

print("="*80)
print("STAGE 1: Environment Setup")
print("="*80)

# 定义路径
KAGGLE_INPUT = Path("/kaggle/input")
PROJECT_ROOT = Path("/MCM")
CONFIG_SRC = None
CODE_SRC = None
DATA_SRC = None

# 1. 自动寻找资源目录
print("🔍 Searching for datasets...")
for d in KAGGLE_INPUT.iterdir():
    d_name = d.name.lower()
    # 找配置
    if "config" in d_name and (d / "experiment_index.json").exists():
        CONFIG_SRC = d
        print(f"  ✓ Configs found: {d}")
    # 找代码
    elif "code" in d_name and (d / "modules").exists():
        CODE_SRC = d
        print(f"  ✓ Code found: {d}")
    # 找数据
    elif "data" in d_name and (d / "downloaded_model").exists():
        DATA_SRC = d
        print(f"  ✓ Data found: {d}")

if not all([CONFIG_SRC, CODE_SRC, DATA_SRC]):
    print("❌ Error: Missing datasets. Please check inputs.")
    # Fallback logic if needed...

# 2. 构建运行环境 /MCM
if not PROJECT_ROOT.exists():
    print(f"🚀 Building project root at {PROJECT_ROOT}...")
    shutil.copytree(CODE_SRC, PROJECT_ROOT, dirs_exist_ok=True)
    
    # 链接数据文件
    print("🔗 Linking data files...")
    for item in DATA_SRC.iterdir():
        if item.name.startswith("."): continue
        target = PROJECT_ROOT / item.name
        if not target.exists():
            if item.is_dir():
                try:
                    shutil.copytree(item, target)
                except:
                    os.symlink(item, target)
            else:
                shutil.copy2(item, target)

# 3. 安装依赖
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
print("📦 Installing dependencies...")
os.system(f"{sys.executable} -m pip install -r requirements_kaggle.txt -q")

# ==========================================
# STAGE 2: Execution Loop
# ==========================================
print("\n" + "="*80)
print("STAGE 2: Running Experiments")
print("="*80)

# 读取索引
with open(CONFIG_SRC / "experiment_index.json") as f:
    index = json.load(f)

for exp in index:
    eid = exp['id']
    if eid < EXP_ID_START or eid >= EXP_ID_END:
        continue
        
    print(f"\n▶️  Running Exp ID {eid}: {exp['seq']} | {exp['dataset']} | {exp['strategy']}")
    
    # 复制配置文件到工作目录
    config_file_name = exp['file']
    src_config = CONFIG_SRC / config_file_name
    local_config = PROJECT_ROOT / "current_task_config.json"
    shutil.copy2(src_config, local_config)
    
    # 运行命令
    start_time = time.time()
    cmd = [
        sys.executable, "-m", "scripts.train_with_zero_shot",
        "--config", str(local_config)
    ]
    
    # 创建日志目录
    log_dir = Path("/kaggle/working") / f"ID{eid}_{exp['seq']}_{exp['dataset']}_{exp['strategy']}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    with open(log_dir / "run.log", "w") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
    
    status = "✅ Success" if proc.returncode == 0 else "❌ Failed"
    duration = (time.time() - start_time) / 60
    print(f"   Status: {status} (Time: {duration:.1f} min)")
    
    if proc.returncode != 0:
        print(f"   ⚠️ Check logs at: {log_dir}/run.log")
        os.system(f"tail -n 20 {log_dir}/run.log")

# ==========================================
# STAGE 3: Pack Results
# ==========================================
print("\n📦 Packing results...")
zip_name = f"results_ID{EXP_ID_START}_to_{EXP_ID_END}"
shutil.make_archive(f"/kaggle/working/{zip_name}", 'zip', root_dir="/kaggle/working")
print(f"✓ Done. Download {zip_name}.zip")