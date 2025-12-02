# Kaggle 快速回归部署指南（分离模式，200样本/epoch=2）

严格遵循 Kaggle 限制：先检查分离数据集，再复制到可写目录，安装依赖，逐个运行现成配置（不可重新生成，不用 run_all.sh），输出在 `/kaggle/working` 并打包下载。

## 1. 环境检查与复制（Notebook Cell 1）
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

## 2. 安装依赖（Cell 2）
```bash
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

## 3. GPU/设备检查（Cell 3，可选）
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
```

## 4. 运行所有配置（Cell 4）
> 配置已随 mcm-code 提供，无需再运行生成脚本；**不要用 run_all.sh**。确保输出写在 `/kaggle/working`。
```python
import subprocess, sys
from pathlib import Path
root = Path('/MCM')
configs = sorted(root.glob('scripts/configs/kaggle_quick_test/*/*.json'))
print('待运行配置数:', len(configs))
for cfg in configs:
    print('\n=== Running', cfg)
    subprocess.check_call([sys.executable, '-m', 'scripts.train_with_zero_shot', '--config', str(cfg)], cwd=root)

```

## 5. 收集与打包（Cell 5）
```python
import shutil
from pathlib import Path

print("="*80)
print("打包实验结果")
print("="*80)

output_dir = Path("/kaggle/working/checkpoints")
output_zip = Path("/kaggle/working/quicktest.zip")

if output_dir.exists():
    print("\n📦 正在打包...")
    print(f"  源目录: {output_dir}")
    print(f"  目标文件: {output_zip}")
  
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
    else:
        print("❌ 打包失败")
else:
    print("❌ 输出目录不存在，无内容可打包")

print("\n" + "="*80)
print("⚠️ 为节省GPU配额，请完成以下操作:")
print("="*80)
print("  1. 下载 quicktest.zip")
print("  2. 点击右上角 'Stop Session' 停止Notebook")
print("  3. 或等待脚本自动退出后手动停止")
print("="*80)

print("\n✅ 打包完成")

```

## 6. 说明
- 任务序列固定：masc(text)→mate(text)→mner(text)→mabsa(text)→masc(mm)→mate(mm)→mner(mm)→mabsa(mm)。
- 配置包含共享头方案（text/mm 同任务共享 head_key，共4头）与 `none_8heads` 独立头对照。
- 若显存不足，可在 Cell 4 过滤 `configs` 仅跑部分方法。 
