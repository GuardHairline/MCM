# Head策略对比实验 (4-Head vs 8-Head)

## 📌 实验目标
在 **8步任务序列** (Text-Only x4 -> Multimodal x4) 中，对比 **共享Head** (4头) 与 **独立Head** (8头) 的性能差异。
- **序列**: `masc`->`mate`->`mner`->`mabsa` (Text) -> `masc`->`mate`->`mner`->`mabsa` (Multi)
- **配置**: CRF=True, BiLSTM=False, Strategy=None
- **机制**: 上一步的 `output_model` 自动作为下一步的 `pretrained_model`，且所有步骤共享同一个 `train_info.json` 记录完整曲线。

---
## 🔧 步骤 1：环境初始化与项目部署

**重要**：Kaggle 的 input 目录是只读的。我们需要将代码和数据复制到系统根目录 `/MCM` 下运行。

### Cell 1: 检查环境与复制项目

```python
import os
import sys
import shutil
from pathlib import Path

print("="*80)
print("CRF修复测试 - 环境检查")
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

print("\n✅ 项目准备完成")

```
## 📦 步骤 2：安装依赖与硬件检查
### Cell 2: 安装依赖
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


# 检查训练循环
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
## 🚀 步骤 4：运行实验
选择运行实验 1 (4-Head)、实验 2 (8-Head) 或全部运行。

### Cell 4: 执行训练
```python
import json
import subprocess
import time
import os
import sys

# === 实验选择 ===
# "1": 4-Head (复用)
# "2": 8-Head (独立)
# "all": 运行两者
RUN_ID = "1" 

# 确保在 /MCM 目录下
WORK_DIR = "/MCM"
if os.getcwd() != WORK_DIR:
    os.chdir(WORK_DIR)
    sys.path.insert(0, WORK_DIR)
    
# 读取索引
index_path = f"{WORK_DIR}/scripts/configs/kaggle_head_comparison/experiment_index.json"
if not os.path.exists(index_path):
    print(f"❌ 索引文件未找到: {index_path}")
    # 备用：尝试直接查找文件
    configs_to_run = []
else:
    with open(index_path, 'r') as f:
        exp_index = json.load(f)

    # 确定要运行的列表
    configs_to_run = []
    if RUN_ID == "all":
        configs_to_run = list(exp_index.values())
    elif str(RUN_ID) in exp_index:
        configs_to_run = [exp_index[str(RUN_ID)]]
    else:
        print("❌ 无效的选择")

# 开始循环
for config_rel_path in configs_to_run:
    # 转换为绝对路径
    # 注意：如果索引里已经是绝对路径则不用拼接，这里做个兼容处理
    if config_rel_path.startswith("/"):
        config_path = config_rel_path
    else:
        config_path = f"{WORK_DIR}/{config_rel_path}"
    
    print(f"\n{'='*60}")
    print(f"🚀 开始实验: {os.path.basename(config_path)}")
    print(f"📄 配置文件: {config_path}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # 修正点：使用 train_with_zero_shot 模块来运行序列配置
    # -m scripts.train_with_zero_shot 确保 Python 能够正确解析包路径
    cmd = f"python -m scripts.train_with_zero_shot --config {config_path}"
    
    print(f"执行命令: {cmd}")
    
    # 在 /MCM 目录下运行，并实时输出日志
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        cwd=WORK_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # 实时打印输出，防止 Kaggle 认为进程卡死
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    duration = (time.time() - start_time) / 60
    
    if process.returncode == 0:
        print(f"\n✅ 实验完成 (耗时: {duration:.1f} 分钟)")
    else:
        print(f"\n❌ 实验失败 (返回码: {process.returncode})")
```
## 📊 步骤 5：结果对比与打包
训练完成后，所有结果（日志、train_info.json）都在 /kaggle/working/output 下。

### Cell 5: 简单结果分析
```python
import json
import pandas as pd
import glob

output_root = "/kaggle/working/output"
results = []

# 查找所有 train_info.json
for info_file in glob.glob(f"{output_root}/**/train_info.json", recursive=True):
    exp_name = Path(info_file).parent.name
    try:
        with open(info_file, 'r') as f:
            data = json.load(f)
            # 获取最后一个 Session (step 8) 的结果
            if "sessions" in data and len(data["sessions"]) > 0:
                last_session = data["sessions"][-1]
                metrics = last_session.get("details", {}).get("final_test_metrics", {})
                f1 = metrics.get("chunk_f1", 0)
                results.append({"Experiment": exp_name, "Final_F1": f1})
    except:
        pass

if results:
    print("📊 最终结果对比:")
    print(pd.DataFrame(results))
else:
    print("暂无结果或读取失败")
```
## Cell 6: 打包下载
```python
import shutil
from datetime import datetime

# 打包 output 目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
archive_name = f"Head_Comparison_{timestamp}"
archive_path = f"/kaggle/working/{archive_name}"

print("📦 正在打包...")
shutil.make_archive(archive_path, 'zip', root_dir='/kaggle/working', base_dir='output')

print(f"✅ 打包完成: {archive_name}.zip")
print("👉 请前往右侧 'Output' 面板下载。")
```