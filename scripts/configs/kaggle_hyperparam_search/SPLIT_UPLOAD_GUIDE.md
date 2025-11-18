# Kaggle分离上传指南

## 🎯 为什么使用分离上传？

**问题**：
- 完整项目包含大量数据（data、downloaded_model），几GB大小
- 每次修改代码都要重新上传整个项目
- 上传耗时长，不便于快速迭代

**解决方案**：
- **分离上传**：将项目分为代码和数据两部分
- **mcm-data**：data + downloaded_model（上传一次，几乎不变）
- **mcm-code**：纯代码（频繁修改，快速上传）

**优势**：
- ✅ 首次上传data后，只需更新代码
- ✅ 代码包很小（几MB），上传快（<1分钟）
- ✅ 节省时间和带宽
- ✅ 便于调试和快速迭代

---

## 📦 方案对比

| 模式 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **完整模式** | 简单，一个数据集 | 每次都要上传几GB | 首次使用，不常修改 |
| **分离模式** | 代码更新快 | 需要两个数据集 | 频繁修改代码 |

---

## 🚀 完整流程

### 步骤1：首次上传数据（一次性）

#### 1.1 打包数据

```bash
cd /path/to/MCM
bash scripts/configs/kaggle_hyperparam_search/prepare_data_only.sh
```

会生成 `MCM_data.zip`（约2-5GB）

**包含内容**：
```
MCM_data.zip
├── data/
│   ├── twitter2015_images/
│   ├── twitter2017_images/
│   ├── MNER/
│   ├── MNRE/
│   └── MASC/
└── downloaded_model/
    ├── deberta-v3-base/
    └── vit-base-patch16-224-in21k/
```

#### 1.2 上传到Kaggle

1. 访问 https://www.kaggle.com/datasets
2. 点击 **"New Dataset"**
3. 上传 `MCM_data.zip`
4. 设置：
   - **Title**: MCM Data
   - **Slug**: `mcm-data` ⚠️ 必须是这个名称
   - **Visibility**: Private
5. 点击 **"Create"**
6. 等待上传和解压（10-30分钟）

**验证**：上传后，数据集结构应该是：
```
/kaggle/input/mcm-data/
├── data/
└── downloaded_model/
```

---

### 步骤2：打包并上传代码（首次）

#### 2.1 打包代码

```bash
cd /path/to/MCM
bash scripts/configs/kaggle_hyperparam_search/prepare_code_only.sh
```

会生成 `MCM_code.zip`（约10-50MB）

**包含内容**：
```
MCM_code.zip
├── scripts/
├── models/
├── datasets/
├── continual/
├── modules/
├── utils/
├── requirements.txt
├── requirements_kaggle.txt
└── ...其他代码文件
```

**排除内容**：
- ❌ data/
- ❌ downloaded_model/
- ❌ checkpoints/
- ❌ log/
- ❌ __pycache__/

#### 2.2 上传到Kaggle

1. 访问 https://www.kaggle.com/datasets
2. 点击 **"New Dataset"**
3. 上传 `MCM_code.zip`
4. 设置：
   - **Title**: MCM Code
   - **Slug**: `mcm-code` ⚠️ 必须是这个名称
   - **Visibility**: Private
5. 点击 **"Create"**
6. 等待上传和解压（1-5分钟）

**验证**：上传后，数据集结构应该是：
```
/kaggle/input/mcm-code/
├── scripts/
├── models/
├── datasets/
└── ...
```

---

### 步骤3：创建Notebook

#### 3.1 新建Notebook

1. 访问 https://www.kaggle.com/code
2. New Notebook → Python
3. 标题：`MCM Hyperparameter Search - Split Mode`
4. Accelerator：**GPU P100**

#### 3.2 添加数据集

在Notebook右侧 **Data** 面板：
1. 点击 **"Add Data"**
2. 选择 **"Your Datasets"**
3. 添加 `mcm-code` ✅
4. 再次点击 **"Add Data"**
5. 添加 `mcm-data` ✅

**确认**：现在应该有两个数据集

#### 3.3 运行Notebook

使用以下Cell代码（脚本会自动检测分离模式）：

**Cell 1 - 环境检查**：
```python
import os
import sys
import shutil
from pathlib import Path

print("="*80)
print("检查数据集")
print("="*80)

print("\n可用数据集:")
for dataset in os.listdir("/kaggle/input"):
    print(f"  - {dataset}")

# 确认两个数据集都存在
assert os.path.exists("/kaggle/input/mcm-code"), "❌ 缺少 mcm-code 数据集"
assert os.path.exists("/kaggle/input/mcm-data"), "❌ 缺少 mcm-data 数据集"

print("\n✓ 两个数据集都已添加")
```

**Cell 2 - 复制项目**（脚本自动处理）：
```python
# 复制到工作目录
work_dir = Path("/kaggle/working/MCM")

if not work_dir.exists():
    print("复制代码...")
    shutil.copytree("/kaggle/input/mcm-code", work_dir)
    print("✓ 代码复制完成")
else:
    print("✓ 工作目录已存在")

# 链接数据目录
data_link = work_dir / "data"
model_link = work_dir / "downloaded_model"

if not data_link.exists():
    print("链接数据目录...")
    os.symlink("/kaggle/input/mcm-data/data", data_link)
    print("✓ data 已链接")

if not model_link.exists():
    print("链接模型目录...")
    os.symlink("/kaggle/input/mcm-data/downloaded_model", model_link)
    print("✓ downloaded_model 已链接")

os.chdir(work_dir)
sys.path.insert(0, str(work_dir))
print(f"\n✓ 工作目录: {os.getcwd()}")
```

**Cell 3-8**：使用 `QUICK_START.md` 中的其他Cell

**或者使用自动脚本**：
```python
# 使用kaggle_runner.py（自动检测分离模式）
runner = work_dir / "scripts/configs/kaggle_hyperparam_search/kaggle_runner.py"
!python {str(runner)} --start_exp 1 --end_exp 5
```

---

### 步骤4：修改代码后重新上传（常用）

每次修改代码后：

#### 4.1 重新打包代码

```bash
cd /path/to/MCM
bash scripts/configs/kaggle_hyperparam_search/prepare_code_only.sh
```

生成新的 `MCM_code.zip`（很快，<1分钟）

#### 4.2 更新Kaggle数据集

1. 访问 https://www.kaggle.com/datasets
2. 找到你的 **mcm-code** 数据集
3. 点击 **"New Version"** （新版本）
4. 上传新的 `MCM_code.zip`
5. 写版本说明：`v2: 修复bug XXX` 或 `v3: 添加功能YYY`
6. 点击 **"Create"**
7. 等待更新（1-3分钟）

#### 4.3 在Notebook中使用新版本

**方法1**：Notebook会自动使用最新版本（推荐）

**方法2**：手动选择版本
1. 在Notebook的Data面板
2. 点击mcm-code数据集右侧的齿轮
3. 选择最新版本

**验证**：
```python
# 检查版本
!ls -la /kaggle/input/mcm-code/.kaggle/
```

---

## 📊 工作流程图

```
首次设置:
┌─────────────────┐
│ prepare_data.sh │ → MCM_data.zip → Kaggle: mcm-data (一次性)
└─────────────────┘

┌─────────────────┐
│ prepare_code.sh │ → MCM_code.zip → Kaggle: mcm-code (首次)
└─────────────────┘

┌─────────────────┐
│ Notebook + 两数据集│ → 运行实验
└─────────────────┘

后续修改代码:
┌─────────────────┐
│ 修改代码        │
└─────────────────┘
         ↓
┌─────────────────┐
│ prepare_code.sh │ → MCM_code.zip
└─────────────────┘
         ↓
┌─────────────────┐
│ 更新mcm-code v2 │ (New Version)
└─────────────────┘
         ↓
┌─────────────────┐
│ Notebook自动用新版│ → 运行实验
└─────────────────┘
```

---

## 🔍 验证清单

### 数据集验证

- [ ] `mcm-data` 数据集存在
- [ ] `/kaggle/input/mcm-data/data/` 有内容
- [ ] `/kaggle/input/mcm-data/downloaded_model/` 有内容

- [ ] `mcm-code` 数据集存在
- [ ] `/kaggle/input/mcm-code/scripts/` 有内容
- [ ] `/kaggle/input/mcm-code/models/` 有内容

### Notebook验证

- [ ] 两个数据集都已添加
- [ ] GPU P100已选择
- [ ] Cell 1能检测到两个数据集
- [ ] Cell 2成功复制和链接
- [ ] `/kaggle/working/MCM/data/` 可访问
- [ ] `/kaggle/working/MCM/downloaded_model/` 可访问

---

## ⚠️ 常见问题

### 问题1：只添加了mcm-code，忘记添加mcm-data

**现象**：
```
FileNotFoundError: data/twitter2015_images/
```

**解决**：
在Notebook中添加 `mcm-data` 数据集

### 问题2：符号链接失败

**现象**：
```
OSError: symbolic link privilege not held
```

**解决**：
脚本会自动回退到复制模式（较慢但有效）

### 问题3：数据集名称错误

**现象**：
```
未找到项目路径
```

**解决**：
确保数据集slug必须是：
- `mcm-code` （不是 mcm_code 或 MCM-Code）
- `mcm-data` （不是 mcm_data 或 MCM-Data）

### 问题4：代码更新后Notebook还是旧版本

**解决**：
1. 检查mcm-code是否更新成功
2. 重启Notebook（Kernel → Restart）
3. 或手动选择最新版本

---

## 💡 最佳实践

### 1. 版本管理

为代码数据集添加有意义的版本说明：
- ❌ `v2`
- ✅ `v2: 修复依赖冲突问题`
- ✅ `v3: 添加分离模式支持`

### 2. 打包前检查

```bash
# 确认在项目根目录
pwd

# 检查data目录
ls -lh data/

# 检查downloaded_model
ls -lh downloaded_model/
```

### 3. 只在必要时更新数据

`mcm-data` 只在以下情况需要更新：
- ✅ 添加了新数据集
- ✅ 更新了预训练模型
- ❌ 修改了代码（不需要）
- ❌ 修改了配置（不需要）

### 4. 利用Kaggle版本控制

- 每次上传代码时写清楚版本说明
- 可以回退到任何历史版本
- 便于追踪问题

---

## 📈 效率对比

| 操作 | 完整模式 | 分离模式 | 节省 |
|------|---------|---------|------|
| 首次上传 | 上传1个数据集（5GB） | 上传2个数据集（5GB） | 相同 |
| 修改代码后 | 重新上传5GB | 只上传50MB | **99%** |
| 上传时间 | 20-30分钟 | 1-3分钟 | **90%** |
| 迭代速度 | 慢 | 快 | **10倍** |

---

## 📝 总结

**分离上传的核心优势**：
1. ✅ 数据只上传一次
2. ✅ 代码更新超快（<3分钟）
3. ✅ 便于快速调试和迭代
4. ✅ 节省带宽和时间

**适用场景**：
- ✅ 需要频繁修改代码
- ✅ 数据集不常变化
- ✅ 快速调试和实验

**使用建议**：
- 首次使用可以用完整模式熟悉流程
- 开始频繁修改后切换到分离模式
- 两种模式的Notebook代码完全兼容

---

现在开始：

```bash
# 1. 打包数据（一次性）
bash scripts/configs/kaggle_hyperparam_search/prepare_data_only.sh

# 2. 打包代码
bash scripts/configs/kaggle_hyperparam_search/prepare_code_only.sh

# 3. 按照步骤上传到Kaggle
```

Good luck! 🚀


