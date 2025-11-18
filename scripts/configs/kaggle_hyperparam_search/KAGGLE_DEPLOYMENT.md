# Kaggle部署指南 - MCM项目超参数搜索

本指南详细说明如何在Kaggle上运行MCM项目的超参数搜索实验。

## 📋 目录

1. [前期准备](#前期准备)
2. [项目打包上传](#项目打包上传)
3. [创建Kaggle Notebook](#创建kaggle-notebook)
4. [运行实验](#运行实验)
5. [结果下载](#结果下载)
6. [注意事项](#注意事项)
7. [故障排查](#故障排查)

---

## 🔧 前期准备

### 1. 检查项目结构

确保你的项目包含以下关键文件：
- `requirements.txt` - Python依赖
- `scripts/train_with_zero_shot.py` - 训练脚本
- `scripts/configs/kaggle_hyperparam_search/` - 配置文件
- 所有必要的代码文件（`models/`, `datasets/`, `continual/` 等）

### 2. 准备数据集

确保以下数据在项目中：
- `data/twitter2015_images/` - Twitter2015图片数据
- `data/MNER/` - MNER数据集
- `data/MNRE/` - MNRE数据集  
- `data/MASC/` - MASC数据集
- `data/MABSA/` - MABSA数据集（如果有）
- `downloaded_model/` - 预训练模型（DeBERTa, ViT等）

### 3. 清理不必要文件

为了减小上传大小，删除以下内容：
```bash
bash scripts/configs/kaggle_hyperparam_search/prepare_for_kaggle.sh
```

这会自动：
- 删除 `__pycache__/` 和 `.pyc` 文件
- 删除已有的 `checkpoints/` （结果会在Kaggle上重新生成）
- 删除 `.git/` 目录（如果需要）
- 压缩项目为 `MCM_kaggle.zip`

---

## 📦 项目打包上传

### 方法1：使用准备脚本（推荐）

```bash
# 运行准备脚本
cd scripts/configs/kaggle_hyperparam_search
bash prepare_for_kaggle.sh

# 脚本会生成 MCM_kaggle.zip
```

### 方法2：手动打包

```bash
# 在项目根目录
cd /path/to/MCM

# 清理缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete

# 打包（排除不必要文件）
zip -r MCM_kaggle.zip . \
    -x "*.git*" \
    -x "*__pycache__*" \
    -x "*.pyc" \
    -x "*checkpoints/*" \
    -x "*.zip"
```

### 上传到Kaggle数据集

1. 访问 [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
2. 点击 **"New Dataset"**
3. 上传 `MCM_kaggle.zip`
4. 设置数据集名称：`mcm-project` （或你喜欢的名称）
5. 选择 **Private** （私有数据集）
6. 点击 **"Create"**

⚠️ **注意**：Kaggle数据集上传后会自动解压，所以你的项目文件会在 `/kaggle/input/mcm-project/MCM/` 或 `/kaggle/input/mcm-project/` 下。

---

## 📓 创建Kaggle Notebook

### 1. 创建新Notebook

1. 访问 [https://www.kaggle.com/code](https://www.kaggle.com/code)
2. 点击 **"New Notebook"**
3. 选择 **Python**
4. 设置Notebook标题：`MCM Hyperparameter Search`

### 2. 配置Notebook设置

点击右侧设置面板：

**加速器 (Accelerator)**：
- 选择 **GPU P100** （推荐）
- 或 **GPU T4** （如果P100不可用）
- ⚠️ 不要选择 TPU

**持久化 (Persistence)**：
- 如果可用，开启 **"Enable GPU"** 和 **"Internet"**

**数据集 (Data)**：
- 点击 **"Add Data"**
- 搜索并添加你上传的数据集：`mcm-project`
- 数据集会挂载到 `/kaggle/input/mcm-project/`

### 3. Notebook代码

在第一个Cell中粘贴以下代码：

```python
# Cell 1: 环境设置和项目复制
import os
import sys
import shutil
from pathlib import Path

# 检查项目路径
print("检查数据集路径...")
print("可用数据集:", os.listdir("/kaggle/input"))

# 找到项目路径
dataset_name = "mcm-project"
possible_paths = [
    f"/kaggle/input/{dataset_name}/MCM",
    f"/kaggle/input/{dataset_name}",
]

project_source = None
for path in possible_paths:
    if os.path.exists(path):
        project_source = Path(path)
        print(f"✓ 找到项目: {path}")
        break

if project_source is None:
    raise FileNotFoundError("未找到MCM项目！")

# 复制到工作目录
work_dir = Path("/kaggle/working/MCM")
if not work_dir.exists():
    print("复制项目到工作目录...")
    shutil.copytree(project_source, work_dir)
    print("✓ 复制完成")

# 切换工作目录
os.chdir(work_dir)
sys.path.insert(0, str(work_dir))
print(f"当前工作目录: {os.getcwd()}")
```

```python
# Cell 2: 安装依赖
!pip install -q transformers datasets torch torchvision pillow numpy pandas scikit-learn matplotlib seaborn tqdm

print("✓ 依赖安装完成")
```

```python
# Cell 3: 检查GPU
import torch

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ 未检测到GPU")
```

```python
# Cell 4: 运行实验
# 使用kaggle_runner.py脚本

# 从数据集中复制运行脚本
runner_script = work_dir / "scripts/configs/kaggle_hyperparam_search/kaggle_runner.py"

if not runner_script.exists():
    print(f"错误: 运行脚本不存在: {runner_script}")
else:
    # 运行前5个实验（根据时间调整）
    !python {str(runner_script)} --start_exp 1 --end_exp 5
```

### 4. 调整实验范围

根据Kaggle时间限制调整实验数量：

| GPU类型 | 可用时间 | 建议实验数 |
|---------|----------|-----------|
| P100    | 9小时    | 3-5个实验  |
| T4      | 9小时    | 2-3个实验  |

**估算**：每个实验约1.5-2小时（取决于任务和数据集大小）

---

## 🚀 运行实验

### 方式1：运行全部Cell（推荐）

点击 **"Run All"** 按钮

### 方式2：逐个Cell运行

依次点击每个Cell的运行按钮

### 监控进度

- 观察输出日志
- 检查 `/kaggle/working/checkpoints/` 目录
- 查看 `experiment_progress.json` 了解进度

### 分批运行策略

由于Kaggle有9-12小时时间限制，建议分批运行：

**第1批**（实验1-5）：
```python
!python kaggle_runner.py --start_exp 1 --end_exp 5
```

**第2批**（实验6-10）：
```python
!python kaggle_runner.py --start_exp 6 --end_exp 10
```

每批运行完成后：
1. 下载 `/kaggle/working/checkpoints/` 到本地
2. 创建新的Notebook继续下一批

---

## 💾 结果下载

### 下载检查点文件

在Notebook的最后一个Cell中：

```python
# 打包结果
import shutil

output_dir = Path("/kaggle/working/checkpoints")
if output_dir.exists():
    shutil.make_archive("/kaggle/working/results", 'zip', output_dir)
    print("✓ 结果已打包: /kaggle/working/results.zip")
    print(f"  大小: {(Path('/kaggle/working/results.zip').stat().st_size / 1e6):.1f} MB")
```

然后点击右侧 **Output** 标签页，下载 `results.zip`

### 下载单个文件

也可以在Notebook中直接查看和下载单个文件：

```python
# 列出所有结果文件
!ls -lh /kaggle/working/checkpoints/
```

---

## ⚠️ 注意事项

### Kaggle限制

1. **运行时间**: 9-12小时后会自动终止
   - 解决：分批运行，每批3-5个实验
   
2. **磁盘空间**: ~20GB
   - 解决：定期删除中间结果，只保留最终模型

3. **GPU显存**: P100约16GB
   - 解决：如果OOM，减小batch_size

4. **网络限制**: 某些外部资源可能无法访问
   - 解决：将预训练模型包含在数据集中

### 路径问题

- Kaggle数据集是**只读**的（`/kaggle/input/`）
- 所有输出必须写到 `/kaggle/working/`
- 项目代码层级不能超过5层（已通过复制到工作目录解决）

### 模型保存

配置文件已自动将checkpoint路径设置为：
```
/kaggle/working/checkpoints/
```

不需要手动修改。

---

## 🔍 故障排查

### 问题1: ModuleNotFoundError: No module named 'scripts'

**原因**: 工作目录不正确

**解决**:
```python
import os, sys
os.chdir("/kaggle/working/MCM")
sys.path.insert(0, "/kaggle/working/MCM")
```

### 问题2: FileNotFoundError: 数据集路径不存在

**原因**: 数据集未正确挂载

**解决**:
```python
# 检查数据集
!ls -la /kaggle/input/
!ls -la /kaggle/input/mcm-project/
```

### 问题3: CUDA out of memory

**原因**: GPU显存不足

**解决**:
1. 修改配置文件中的 `batch_size`
2. 或在训练脚本中添加：
```python
torch.cuda.empty_cache()
```

### 问题4: 运行时间超过限制

**原因**: Kaggle 9小时限制

**解决**:
- 减少每批实验数量
- 使用 `--start_exp` 和 `--end_exp` 参数分批运行

### 问题5: 无法保存结果

**原因**: 写入只读目录

**解决**:
确保所有输出路径都在 `/kaggle/working/` 下

---

## 📊 结果分析

下载结果后，在本地运行分析脚本：

```bash
# 解压结果
unzip results.zip -d ./kaggle_results

# 运行分析
python scripts/configs/kaggle_hyperparam_search/analyze_kaggle_results.py \
    --results_dir ./kaggle_results
```

---

## 📞 获取帮助

遇到问题？

1. 检查Kaggle Notebook的输出日志
2. 查看 `/kaggle/working/experiment_progress.json`
3. 检查数据集是否正确上传
4. 确认GPU是否可用

---

## 实验配置总结

- **总实验数**: 39
- **任务**: MATE, MNER, MABSA
- **每个任务**: text_only → multimodal
- **超参数**: lr, step_size, gamma
- **每批建议数**: 5
- **预计总时间**: 约 12 个Kaggle会话

---

Good luck! 🚀
