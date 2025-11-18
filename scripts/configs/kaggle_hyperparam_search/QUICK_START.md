# Kaggle快速开始指南

## 🚀 5分钟上手

### 步骤1：本地准备（2分钟）

```bash
cd /path/to/MCM

# 1. 生成配置（如果还没生成）
python scripts/generate_kaggle_hyperparameter_configs.py

# 2. 打包项目
bash scripts/configs/kaggle_hyperparam_search/prepare_for_kaggle.sh
# 按提示选择：删除checkpoints(y)、保留log(n)、保留.git(n)
# 生成 MCM_kaggle.zip
```

### 步骤2：上传Kaggle（10分钟）

1. 访问 https://www.kaggle.com/datasets
2. New Dataset → 上传 `MCM_kaggle.zip`
3. 名称：`mcm-project`，私有
4. Create（等待解压，约5-10分钟）

### 步骤3：创建Notebook（3分钟）

1. 访问 https://www.kaggle.com/code  
2. New Notebook → Python
3. 设置：Accelerator = **GPU P100**
4. Add Data → 选择 `mcm-project`

### 步骤4：复制代码运行

**Cell 1 - 环境检查**：
```python
import os, sys, shutil
from pathlib import Path

# 找到项目
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

if not project_source:
    raise FileNotFoundError("未找到MCM项目")

# 复制到工作目录
work_dir = Path("/kaggle/working/MCM")
if not work_dir.exists():
    print("复制项目...")
    shutil.copytree(project_source, work_dir)
    
os.chdir(work_dir)
sys.path.insert(0, str(work_dir))
print(f"✓ 工作目录: {os.getcwd()}")
```

**Cell 2 - 安装依赖**：
```python
# 优先使用Kaggle优化版本
!pip install -q -r requirements_kaggle.txt 2>/dev/null || \
 pip install -q pytorch_crf sentencepiece protobuf==3.20.3

print("✓ 依赖安装完成（忽略版本警告）")
```

**Cell 3 - 检查GPU**：
```python
import torch
assert torch.cuda.is_available(), "GPU不可用"
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
```

**Cell 4 - 运行实验**：
```python
# 设置实验范围（建议3-5个实验）
START_EXP = 1
END_EXP = 5

# 运行
runner = work_dir / "scripts/configs/kaggle_hyperparam_search/kaggle_runner.py"
!python {str(runner)} --start_exp {START_EXP} --end_exp {END_EXP}
```

**Cell 5 - 打包结果**：
```python
import shutil
shutil.make_archive("/kaggle/working/results", 'zip', "/kaggle/working/checkpoints")
print("✓ 结果已打包: /kaggle/working/results.zip")
print("  在右侧Output标签页下载")
```

---

## ⚡ 关键要点

### ✅ 必须做

1. **选GPU P100**（不要选CPU或TPU）
2. **添加数据集**（mcm-project）
3. **复制项目到 /kaggle/working/MCM**
4. **每批3-5个实验**（避免超时）

### ⚠️ 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| 找不到项目 | 未添加数据集 | Add Data → mcm-project |
| ModuleNotFoundError | 工作目录错误 | 确保 os.chdir(work_dir) |
| CUDA out of memory | batch_size太大 | 减小batch_size |
| 版本冲突警告 | Kaggle预装版本 | **忽略**，不影响运行 |

### 📊 时间预估

| 阶段 | 时间 |
|------|------|
| 上传数据集 | 10-20分钟 |
| 创建Notebook | 3分钟 |
| 环境设置 | 5分钟 |
| 运行3个实验 | 4.5-6小时 |
| 运行5个实验 | 7.5-10小时 |

---

## 🔍 验证清单

运行前检查：
- [ ] GPU P100已选择
- [ ] 数据集mcm-project已添加
- [ ] Cell 1成功找到项目
- [ ] Cell 2依赖安装完成
- [ ] Cell 3 GPU可用

运行中监控：
- [ ] 实验进度正常输出
- [ ] GPU使用率50-90%
- [ ] 无OOM错误

运行后验证：
- [ ] /kaggle/working/checkpoints/ 有文件
- [ ] results.zip 已生成
- [ ] 文件大小合理（几百MB）

---

## 📚 详细文档

遇到问题？查看：

1. **[KAGGLE_SETUP_GUIDE.md](KAGGLE_SETUP_GUIDE.md)** - 完整步骤（676行）
2. **[DEPENDENCIES.md](DEPENDENCIES.md)** - 依赖冲突详解
3. **[KAGGLE_DEPLOYMENT.md](KAGGLE_DEPLOYMENT.md)** - 原始部署说明

---

## 💡 提示

1. **分批运行**：
   - Batch 1: 实验1-5（一个Notebook）
   - Batch 2: 实验6-10（新Notebook）
   - ...依次类推

2. **下载策略**：
   - 每批完成后立即下载results.zip
   - 本地合并所有批次结果
   - 统一分析

3. **节省时间**：
   - 使用 requirements_kaggle.txt（快）
   - 不要重复复制项目（检查work_dir.exists()）
   - 监控GPU使用率（应该>50%）

---

Good luck! 🎉


