# Kaggle 超参数搜索快速教程

## 📝 目录

1. [账号准备](#1-账号准备)
2. [代码和数据准备](#2-代码和数据准备)
3. [上传到Kaggle](#3-上传到kaggle)
4. [Notebook配置](#4-notebook配置)
5. [运行模式选择](#5-运行模式选择)
6. [重要注意事项](#6-重要注意事项)
7. [结果下载](#7-结果下载)

---

## 1. 账号准备

### 1.1 注册账号
- 使用**邮箱**或**Google账号**注册
- ⚠️ **一个人只能有一个 Kaggle 账号**（严格限制）

### 1.2 认证流程（必须完成才能使用GPU）

**步骤**：
1. 📧 **邮箱认证**：验证注册邮箱
2. 📱 **手机认证**：绑定手机号码
3. 🎭 **人脸认证**（KYC）：上传身份证明和自拍

**重要提示**：
- ✅ 只有完成全部认证后才能使用 GPU
- ⏱️ 认证通常需要 1-3 个工作日
- 🚫 未认证账号只能使用 CPU

### 1.3 GPU 配额

认证后可获得：
- **P100 GPU**: 约 30 小时/周
- **T4 GPU**: 约 36 小时/周
- 每周一重置配额

---

## 2. 代码和数据准备

### 2.1 分离打包策略（强烈推荐）🌟

**为什么要分离？**
- 数据文件很大（2-5GB），很少修改
- 代码文件较小（10-50MB），经常修改
- **更新代码只需 1-3 分钟**，而不是 20 分钟

**打包步骤**：

```bash
cd /path/to/MCM

# 1. 打包数据（一次性，以后不用再打包）
bash scripts/configs/kaggle_hyperparam_search/prepare_data_only.sh
# 生成: MCM_data.zip (~2-5GB)

# 2. 打包代码
bash scripts/configs/kaggle_hyperparam_search/prepare_code_only.sh
# 生成: MCM_code.zip (~10-50MB)
```

**修改代码后只需**：
```bash
# 重新打包代码（<1分钟）
bash scripts/configs/kaggle_hyperparam_search/prepare_code_only.sh

# 在 Kaggle 更新 mcm-code 数据集版本（1-3分钟）
```

### 2.2 完整打包（首次使用可选）

```bash
bash scripts/configs/kaggle_hyperparam_search/prepare_for_kaggle.sh
# 生成: MCM_kaggle.zip (~2-5GB)
```

---

## 3. 上传到Kaggle

### 3.1 分离模式上传（推荐）

#### 步骤1：上传数据数据集（一次性）

1. 访问 [Kaggle Datasets](https://www.kaggle.com/datasets)
2. 点击 **"New Dataset"**
3. 上传 `MCM_data.zip`
4. 配置：
   - **Title**: MCM Data
   - **Slug**: `mcm-data` ⚠️ **必须是这个名称**
   - **Visibility**: Private
5. 点击 **"Create"**
6. 等待上传（15-30分钟，**只需一次**）

#### 步骤2：上传代码数据集

1. 再次访问 [Kaggle Datasets](https://www.kaggle.com/datasets)
2. 点击 **"New Dataset"**
3. 上传 `MCM_code.zip`
4. 配置：
   - **Title**: MCM Code
   - **Slug**: `mcm-code` ⚠️ **必须是这个名称**
   - **Visibility**: Private
5. 点击 **"Create"**
6. 等待上传（1-5分钟）

#### 步骤3：修改代码后更新（常用操作）⚡

1. 找到 `mcm-code` 数据集
2. 点击 **"New Version"**（新版本）
3. 上传新的 `MCM_code.zip`
4. 填写版本说明：`v2: 修复XXX bug`
5. 点击 **"Create"**
6. 等待更新（**1-3分钟**）✨

### 3.2 完整模式上传

1. 上传 `MCM_kaggle.zip` 为一个数据集
2. Slug 设为 `mcm-project`
3. 修改代码需要重新上传整个数据集（20分钟）

---

## 4. Notebook配置

### 4.1 创建 Notebook

1. 访问 [Kaggle Code](https://www.kaggle.com/code)
2. 点击 **"New Notebook"**
3. 选择 **Python**
4. 标题：`MCM Hyperparameter Search - Batch 1`

### 4.2 配置 GPU

点击右侧 **Settings** → **Accelerator**:
- ✅ 选择 **GPU P100**（推荐）或 **GPU T4**
- ❌ 不要选择 TPU

### 4.3 添加数据集

**分离模式**：
1. 点击右侧 **Add Data**
2. 搜索并添加 `mcm-code` ✅
3. 再次点击 **Add Data**
4. 搜索并添加 `mcm-data` ✅
5. 确认两个数据集都已添加

**完整模式**：
- 只需添加 `mcm-project`

### 4.4 其他设置

- **Internet**: 开启（安装依赖需要）
- **Environment**: 保持默认
- **Persistence**: ⚠️ **不建议开启**（见下文说明）

---

## 5. 运行模式选择

### 5.1 交互式模式（Interactive Session）

**特点**：
- 📝 可以逐个运行 Cell，实时查看结果
- 💻 需要保持浏览器打开
- 🔴 关闭浏览器后会话结束，虚拟机销毁
- ⚠️ **不稳定**，网页可能崩溃

**适用场景**：
- 调试代码
- 查看中间结果
- 运行时间 < 2 小时

**如何使用**：
1. 直接点击 Cell 左侧的运行按钮
2. 或使用快捷键 `Shift + Enter`

### 5.2 提交模式（Commit & Run）⭐ **推荐**

**特点**：
- 🚀 运行在 Kaggle 服务器，**不依赖浏览器**
- 🌙 可以关闭浏览器，关掉电脑
- ⏰ 只有超时才会中断（9-12小时）
- 🔒 运行过程中不能修改代码
- 📦 结果自动保存到 Output 页面

**如何使用**：
1. 确认所有代码正确
2. 点击右上角 **"Save & Run All"** 或 **"Commit & Run"**
3. 等待运行完成（可以关闭页面）
4. 回来后在 **Output** 标签下载结果

**查看运行状态**：
1. 进入 Notebook 页面
2. 点击右侧 **"Versions"** 标签
3. 最新版本会显示：
   - ⏳ Running（运行中）
   - ✅ Complete（完成）
   - ❌ Failed（失败）

### 5.3 Persistence 设置（不推荐）⚠️

**什么是 Persistence？**
- 在交互式会话中保存 `/kaggle/working` 的文件和变量
- 下次打开 Notebook 时可以继续使用

**为什么不推荐？**
- ❌ 容易导致网页崩溃
- ❌ 占用更多内存
- ❌ 状态不一致问题
- ✅ **用提交模式更稳定可靠**

**如果一定要用**：
1. Settings → Persistence
2. 选择 **"Files only"**（只保存文件）
3. 不要选择 "Variables and Files"（太占内存）

---

## 6. 重要注意事项

### 6.1 Kaggle 目录结构 ⚠️ 核心概念

**只读目录**（不能写入）：
```
/kaggle/input/              # 所有上传的数据集
  ├── mcm-code/             # 代码数据集（只读）
  ├── mcm-data/             # 数据数据集（只读）
  └── ...
```

**可写目录**（可以保存文件）：
```
/kaggle/working/            # 唯一可写的目录
  └── checkpoints/          # 我们的输出目录
      ├── train_info_*.json
      └── *.pt
```

**项目运行目录**（复制后可写）：
```
/MCM/                       # 从 /kaggle/input 复制过来
  ├── scripts/
  ├── models/
  ├── data/                 # 链接到 mcm-data/data
  └── ...
```

### 6.2 文件下载限制 ⚠️

**只能下载**：
- ✅ `/kaggle/working/` 下的**一级文件**
- ✅ 例如：`/kaggle/working/experiment_results.zip`
- ❌ 不能下载子目录：`/kaggle/working/checkpoints/file.json`

**解决方案**（已自动完成）：
- 脚本会自动打包 `/kaggle/working/checkpoints/` 为 `/kaggle/working/experiment_results.zip`
- 直接下载 `experiment_results.zip` 即可

### 6.3 时间限制 ⏰

**Kaggle 限制**：
- 最长运行时间：**9-12 小时**
- 超时会**强制中断**，未保存的结果会丢失

**应对策略**：

**方案1：分批运行（推荐）**
```python
# Batch 1: 实验 1-5（约 7.5-10 小时）
START_EXP = 1
END_EXP = 5

# Batch 2: 实验 6-10
START_EXP = 6
END_EXP = 10
```

**方案2：断点续跑**
- 脚本会自动保存进度到 `experiment_progress.json`
- 中断后重新运行会自动跳过已完成的实验

**每批实验建议数量**：
- P100 GPU: 3-5 个实验（每个约 1.5-2 小时）
- T4 GPU: 2-3 个实验（每个约 2-2.5 小时）

### 6.4 目录层级限制（已解决）

**问题**：
- Kaggle 读取数据集时有目录层级限制（约5级）
- 项目代码层级较深

**解决方案**（脚本自动处理）：
- 脚本会自动将项目复制到根目录 `/MCM`
- 您不需要手动操作

### 6.5 输出路径配置（已自动修复）✅

**问题**：
- 文件必须保存到 `/kaggle/working/checkpoints/`
- 不能保存到 `/MCM/checkpoints/`（无法下载）

**解决方案**（脚本自动处理）：
- `kaggle_runner.py` 会自动更新所有配置文件中的路径
- 实验后会验证文件是否保存到正确位置
- 您会看到详细的日志：
  ```
  [INFO] 正在更新配置文件路径...
  [INFO]   更新路径: train_info_json
  [INFO]     从: checkpoints/train_info_mate_hp1.json
  [INFO]     到: /kaggle/working/checkpoints/train_info_mate_hp1.json
  
  [INFO] ✓ 实验 #1 完成 (耗时: 1.5 小时)
  [INFO]   已保存 15 个文件到 /kaggle/working/checkpoints
  ```

---

## 7. 结果下载

### 7.1 自动打包（已内置）✅

**脚本会自动**：
1. 检查 `/kaggle/working/checkpoints/` 中的所有文件
2. 打包为 `/kaggle/working/experiment_results.zip`
3. 显示文件大小和下载提示

**日志示例**：
```
================================================================================
[INFO] 正在检查并打包实验结果...
================================================================================

[INFO] 检查输出目录: /kaggle/working/checkpoints
[INFO] ✓ 输出目录存在，共 45 个文件

[INFO] 文件列表:
[INFO]   - train_info_mate_hp1.json (125.32 KB)
[INFO]   - twitter2015_mate_none_multimodal_hp1.pt (890.45 MB)
[INFO]   ... 还有 43 个文件

[INFO] 开始打包...
[INFO] ✓ 结果已打包: /kaggle/working/experiment_results.zip
[INFO]   文件大小: 2345.6 MB
```

### 7.2 下载步骤

**交互式模式**：
1. 等待所有实验完成
2. 点击右侧 **"Output"** 标签
3. 找到 `experiment_results.zip`
4. 点击下载图标

**提交模式（Commit）**：
1. 等待 Commit 完成（可以关闭页面）
2. 回到 Notebook，点击 **"Versions"** 标签
3. 选择最新完成的版本
4. 点击版本号进入详情页
5. 在 **"Output"** 标签下载 `experiment_results.zip`

### 7.3 节省 GPU 配额 ⚠️ **非常重要**

**问题**：
- 实验完成后，如果不停止 Session，**会持续消耗 GPU 时间**
- 即使代码已经运行完毕，GPU 仍在计费

**解决方案**：

**交互式模式**：
```
🎉 所有任务已完成！
📦 结果已打包，请下载 experiment_results.zip

⚠️  为节省GPU配额，请立即执行以下操作：
   1. 在右侧 'Output' 标签下载 experiment_results.zip
   2. 点击右上角 'Stop Session' 按钮停止Notebook
   3. 或者等待此脚本自动退出后手动停止
```

**提交模式**：
- ✅ 运行完成后会自动停止，不会浪费配额
- ✅ 这是另一个推荐使用提交模式的理由

**检查剩余配额**：
1. 点击右上角头像
2. 选择 **"Settings"**
3. 查看 **"GPU Quota"**

---

## 📋 完整工作流程 Checklist

### 首次使用

- [ ] 完成账号认证（邮箱 + 手机 + 人脸）
- [ ] 本地打包数据：`prepare_data_only.sh`
- [ ] 本地打包代码：`prepare_code_only.sh`
- [ ] 上传 `mcm-data` 数据集（一次性）
- [ ] 上传 `mcm-code` 数据集
- [ ] 创建 Notebook
- [ ] 配置 GPU P100
- [ ] 添加两个数据集（mcm-code + mcm-data）
- [ ] 设置实验范围（建议 3-5 个）
- [ ] **使用提交模式运行**（Commit & Run）⭐
- [ ] 等待完成（可以关闭页面）
- [ ] 下载 `experiment_results.zip`

### 修改代码后（常用）

- [ ] 本地修改代码
- [ ] 重新打包代码：`prepare_code_only.sh`（<1分钟）
- [ ] 更新 `mcm-code` 数据集版本（1-3分钟）
- [ ] 在 Notebook 的 Data 面板选择最新版本
- [ ] 重新运行（Commit & Run）
- [ ] 下载新结果

### 交互式调试（可选）

- [ ] 使用交互式模式
- [ ] 运行几个 Cell 测试
- [ ] 确认无误后切换到提交模式
- [ ] **记得手动停止 Session** ⚠️

---

## ⚡ 效率对比

| 场景 | 完整模式 | 分离模式 | 节省时间 |
|------|---------|---------|---------|
| **首次上传** | 20 分钟 | 25 分钟 | -5 分钟 |
| **修改代码 1 次** | 20 分钟 | 3 分钟 | **17 分钟** ⚡ |
| **修改代码 5 次** | 100 分钟 | 15 分钟 | **85 分钟** ⚡⚡ |
| **修改代码 10 次** | 200 分钟 | 30 分钟 | **170 分钟** ⚡⚡⚡ |

**结论**：修改超过 2 次代码，分离模式就更高效！

| 运行模式 | 交互式 | 提交模式 |
|---------|-------|---------|
| **稳定性** | ⚠️ 不稳定 | ✅ 稳定 |
| **需要浏览器** | ✅ 需要 | ❌ 不需要 |
| **GPU 浪费** | ⚠️ 容易忘记关 | ✅ 自动停止 |
| **适用场景** | 调试 | 正式运行 |

**结论**：正式运行实验使用**提交模式**！

---

## 🆘 常见问题

### Q1: 为什么我的 GPU 配额用完了？
**A**: 可能原因：
- 实验完成后忘记停止 Session
- 使用交互式模式且开启了 Persistence

**解决**：
- 使用提交模式（自动停止）
- 交互式模式完成后立即点击 "Stop Session"

### Q2: 为什么下载的结果是空的？
**A**: 可能原因：
- 文件保存到了 `/MCM/checkpoints/` 而不是 `/kaggle/working/checkpoints/`

**解决**：
- 使用最新版 `kaggle_runner.py`（已自动修复）
- 查看日志中的路径更新信息

### Q3: 修改代码后还是运行旧代码？
**A**: 可能原因：
- Notebook 中使用的是旧版本的 `mcm-code` 数据集

**解决**：
1. 在 Data 面板检查 `mcm-code` 版本
2. 选择最新版本
3. 重启 Kernel 或重新 Commit

### Q4: 实验运行 9 小时后被强制中断？
**A**: Kaggle 时间限制

**解决**：
- 减少每批实验数量（3-5个）
- 使用断点续跑功能
- 分多个 Notebook 运行

---

## 📚 相关文档

详细说明请参考：
- 📖 **完整部署指南**: `KAGGLE_SETUP_GUIDE.md`
- 🔧 **修复说明**: `FIX_SUMMARY.md`
- 📦 **分离模式详解**: `SPLIT_UPLOAD_GUIDE.md`
- ⚡ **5分钟快速开始**: `QUICK_START.md`

---

## 🎓 推荐学习路径

```
第 1 天：熟悉 Kaggle
  → 完成账号认证
  → 使用完整模式上传项目
  → 交互式运行 1-2 个实验
  → 熟悉界面和操作

第 2 天：切换分离模式
  → 打包并上传 mcm-data（一次性）
  → 打包并上传 mcm-code
  → 使用提交模式运行 3-5 个实验

第 3 天+：高效迭代
  → 本地修改代码
  → 1 分钟重新打包
  → 3 分钟更新 Kaggle
  → 提交模式运行
  → 关闭电脑，第二天回来下载结果 ✨
```

---

**Good luck with your experiments! 🚀**

**记住最重要的三点**：
1. ⭐ **使用分离模式**（节省时间）
2. ⭐ **使用提交模式运行**（稳定可靠）
3. ⭐ **及时停止 Session**（节省配额）


