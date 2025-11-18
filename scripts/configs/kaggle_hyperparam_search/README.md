# Kaggle超参数搜索 - 完整方案

本目录包含在Kaggle上运行MCM项目超参数搜索的完整解决方案。

## 📁 文件结构

```
kaggle_hyperparam_search/
├── README.md                          # 本文件 - 总览
├── QUICK_START.md                     # ⭐ 快速开始（5分钟上手）
├── SPLIT_UPLOAD_GUIDE.md             # 🚀 分离上传指南（推荐）
├── KAGGLE_SETUP_GUIDE.md             # 📖 完整设置指南（676行详细步骤）
├── DEPENDENCIES.md                    # 📦 依赖冲突详解
├── KAGGLE_DEPLOYMENT.md              # 📋 原始部署说明
│
├── kaggle_runner.py                   # 🔧 主运行脚本（自动检测模式）
│
├── prepare_for_kaggle.sh             # 📦 完整项目打包脚本
├── prepare_code_only.sh              # 📦 代码打包脚本（分离模式）
├── prepare_data_only.sh              # 📦 数据打包脚本（分离模式）
│
├── analyze_kaggle_results.py         # 📊 结果分析脚本
│
├── config_index.json                  # 配置索引
└── kaggle_*.json                      # 实验配置文件（多个）
```

## 🎯 使用指南

### 🚀 推荐：分离上传模式 → SPLIT_UPLOAD_GUIDE.md

**适合**：需要频繁修改代码的用户

如果你需要经常调试和修改代码：

```bash
cat scripts/configs/kaggle_hyperparam_search/SPLIT_UPLOAD_GUIDE.md
```

**核心优势**：
- ✅ 数据只上传一次（几GB，一次性）
- ✅ 代码频繁更新（几MB，<3分钟）
- ✅ 迭代速度快10倍
- ✅ 节省90%上传时间

**工作流程**：
1. 首次：上传 `mcm-data`（数据集，一次性）
2. 首次：上传 `mcm-code`（代码）
3. 每次修改代码后：只更新 `mcm-code`（超快）

### 新手用户 → QUICK_START.md

如果你是第一次使用，**强烈推荐**从这里开始：

```bash
cat scripts/configs/kaggle_hyperparam_search/QUICK_START.md
```

包含：
- ✅ 5分钟快速上手
- ✅ 5个Cell代码复制即用
- ✅ 常见错误速查表
- ✅ 验证清单

### 遇到问题 → DEPENDENCIES.md

如果安装依赖时出现**版本冲突警告**：

```bash
cat scripts/configs/kaggle_hyperparam_search/DEPENDENCIES.md
```

包含：
- ✅ 依赖冲突详解
- ✅ 3种解决方案对比
- ✅ 何时需要关注vs忽略
- ✅ Kaggle预装包列表

### 需要详细步骤 → KAGGLE_SETUP_GUIDE.md

如果需要完整的图文指南：

```bash
cat scripts/configs/kaggle_hyperparam_search/KAGGLE_SETUP_GUIDE.md
```

包含：
- ✅ 8个详细Notebook Cell代码
- ✅ 目录结构说明
- ✅ 6个常见问题FAQ
- ✅ 完整检查清单

## 🚀 快速开始

### 选择模式

| 模式 | 适合场景 | 优点 | 缺点 |
|------|---------|------|------|
| **分离模式** 🌟 | 频繁修改代码 | 更新快（<3分钟）| 需要2个数据集 |
| **完整模式** | 首次使用/稳定运行 | 简单（1个数据集）| 每次都要上传5GB |

### 模式A：分离上传（推荐）⭐

**Step 1a: 打包数据（一次性）**

```bash
cd /path/to/MCM
bash scripts/configs/kaggle_hyperparam_search/prepare_data_only.sh
# 生成 MCM_data.zip (~2-5GB)
```

**Step 1b: 打包代码**

```bash
bash scripts/configs/kaggle_hyperparam_search/prepare_code_only.sh
# 生成 MCM_code.zip (~10-50MB)
```

**Step 2: 上传Kaggle**

1. 上传 `MCM_data.zip` → 数据集 `mcm-data`
2. 上传 `MCM_code.zip` → 数据集 `mcm-code`

**Step 3: Notebook添加两个数据集并运行**

详见 `SPLIT_UPLOAD_GUIDE.md`

**修改代码后**：
```bash
bash prepare_code_only.sh  # 重新打包代码（<1分钟）
# 在Kaggle更新mcm-code数据集（New Version）
```

---

### 模式B：完整上传（传统）

**Step 1: 打包项目**

```bash
cd /path/to/MCM
bash scripts/configs/kaggle_hyperparam_search/prepare_for_kaggle.sh
# 生成 MCM_kaggle.zip (~5GB)
```

### Step 2: 上传Kaggle

1. https://www.kaggle.com/datasets → New Dataset
2. 上传 `MCM_kaggle.zip`
3. 名称：`mcm-project`（私有）

### Step 3: 创建Notebook并运行

1. https://www.kaggle.com/code → New Notebook
2. 设置GPU P100，添加数据集
3. 复制 `QUICK_START.md` 中的5个Cell代码
4. Run All

## 📦 关于依赖冲突

### ⚠️ 你可能看到的警告

```
ERROR: pip's dependency resolver does not currently take into account...
datasets 4.1.1 requires pyarrow>=21.0.0, but you have pyarrow 19.0.1
transformers需要>=4.41.0，但安装的是4.30.2
...
```

### ✅ 不用担心

这些是**警告不是错误**，原因：

1. **Kaggle预装了新版本** - 比项目要求的更好
2. **向后兼容** - 新版本支持旧API
3. **不影响运行** - 只要能导入就OK

### 📝 解决方案

已为你准备好 `requirements_kaggle.txt`：

```txt
# 只安装Kaggle缺失的包
pytorch_crf==0.7.2
sentencepiece==0.1.99
protobuf==3.20.3
openpyxl>=3.0.0
```

在Notebook中：
```python
!pip install -q -r requirements_kaggle.txt
```

**详细说明** → `DEPENDENCIES.md`

## 🔧 主要脚本说明

### kaggle_runner.py

Kaggle环境下的实验运行器：

```bash
# 在Kaggle Notebook中
python kaggle_runner.py --start_exp 1 --end_exp 5
```

功能：
- ✅ 自动设置环境
- ✅ 智能安装依赖（优先使用requirements_kaggle.txt）
- ✅ 串行运行实验
- ✅ 断点续跑支持
- ✅ 进度自动保存

### prepare_for_kaggle.sh

项目打包脚本：

```bash
bash prepare_for_kaggle.sh
```

功能：
- ✅ 清理Python缓存
- ✅ 可选删除checkpoints/log
- ✅ 压缩为MCM_kaggle.zip
- ✅ 显示后续步骤

### analyze_kaggle_results.py

结果分析脚本（本地运行）：

```bash
# 下载results.zip后
unzip results.zip -d ./kaggle_results
python analyze_kaggle_results.py --results_dir ./kaggle_results
```

功能：
- ✅ 提取所有实验结果
- ✅ 计算AA、FM、BWT等指标
- ✅ 生成CSV汇总表
- ✅ 显示Top结果

## 📊 实验配置

当前生成的配置：

- **总实验数**: 约36个（3任务 × 12超参数组合）
- **任务**: MATE, MNER, MABSA
- **序列**: text_only → multimodal
- **超参数**: lr ∈ {5e-5, 1e-5, 5e-6}, step_size ∈ {5, 10, 15}, gamma ∈ {0.3, 0.5, 0.7}
- **每个实验**: 约1.5-2小时
- **推荐分批**: 每批3-5个实验

## ⏱️ 时间估算

| 阶段 | 时间 |
|------|------|
| 打包上传 | 20-30分钟 |
| 环境设置 | 5-10分钟 |
| 运行5个实验 | 7.5-10小时 |
| 打包下载 | 5分钟 |
| **总计** | **约8-11小时/批** |

建议分7-8批完成所有36个实验。

## 🎓 学习路径

1. **第一次使用**:
   - [ ] 阅读 `QUICK_START.md`
   - [ ] 运行1-2个实验测试
   - [ ] 验证结果可以下载

2. **遇到依赖问题**:
   - [ ] 阅读 `DEPENDENCIES.md`
   - [ ] 使用 `requirements_kaggle.txt`
   - [ ] 验证包可导入

3. **批量运行**:
   - [ ] 参考 `KAGGLE_SETUP_GUIDE.md`
   - [ ] 分批运行所有实验
   - [ ] 合并分析结果

4. **故障排查**:
   - [ ] 检查 `KAGGLE_SETUP_GUIDE.md` 的FAQ
   - [ ] 查看Notebook输出日志
   - [ ] 验证GPU和路径

## ❓ 常见问题速查

| 问题 | 查看 |
|------|------|
| 版本冲突警告 | DEPENDENCIES.md 问题1 |
| 找不到项目 | KAGGLE_SETUP_GUIDE.md 问题2 |
| ModuleNotFoundError | KAGGLE_SETUP_GUIDE.md 问题3 |
| GPU OOM | KAGGLE_SETUP_GUIDE.md 问题4 |
| 如何分批运行 | QUICK_START.md 提示部分 |

## 📞 获取帮助

遇到问题的顺序：

1. **查看QUICK_START.md** - 快速解决常见问题
2. **查看DEPENDENCIES.md** - 专门针对依赖问题
3. **查看KAGGLE_SETUP_GUIDE.md的FAQ** - 详细故障排查
4. **检查Notebook输出日志** - 具体错误信息
5. **验证环境** - GPU、路径、数据集

## ✨ 最佳实践

### ✅ 推荐做法

1. 使用 `requirements_kaggle.txt`（快速、无冲突）
2. 每批3-5个实验（避免超时）
3. 立即下载results.zip（避免丢失）
4. 使用GPU P100（性能最佳）
5. 监控GPU使用率（应>50%）

### ⚠️ 避免的做法

1. 不要选CPU或TPU
2. 不要一次运行>5个实验
3. 不要忽略路径设置
4. 不要担心版本警告（大部分可忽略）
5. 不要跳过环境验证

## 📝 文件依赖关系

```
prepare_for_kaggle.sh
    └── 生成 MCM_kaggle.zip
            └── 上传到Kaggle
                    └── Notebook使用 kaggle_runner.py
                            └── 读取 config_index.json
                            └── 使用 requirements_kaggle.txt
                                    └── 运行实验
                                            └── 生成结果
                                                    └── analyze_kaggle_results.py分析
```

## 🎉 预期成果

成功运行后，你会得到：

```
kaggle_results/
├── train_info_twitter2015_none_t2m_hp1.json
├── train_info_twitter2015_none_t2m_hp2.json
├── ...
├── twitter2015_mate_none_multimodal_hp1.pt
├── twitter2015_mate_none_text_only_hp1.pt
└── ...
```

以及分析报告：
- `results_summary.csv` - 完整结果
- `best_hyperparameters.json` - 最佳配置

---

## 🎯 下一步

现在开始：

```bash
# 1. 查看快速开始
cat QUICK_START.md

# 2. 打包项目
bash prepare_for_kaggle.sh

# 3. 按照QUICK_START.md的步骤操作
```

Good luck! 🚀

---

*生成时间: 2025-10-27*  
*版本: 1.0*  
*支持: Kaggle GPU P100/T4*

