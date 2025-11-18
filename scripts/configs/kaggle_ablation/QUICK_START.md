# 快速开始 - Kaggle 消融实验

## 🚀 一分钟上手

### 步骤 1: 在 Kaggle 创建 Notebook

1. 访问 https://www.kaggle.com/code
2. 点击 **"New Notebook"**
3. 标题：`CRF Ablation - Account X` (X = 1-6)

### 步骤 2: 配置 Notebook

**Settings:**
- **Accelerator**: GPU P100 或 T4 ✅
- **Internet**: On ✅
- **Persistence**: 可选

**Data (右侧面板):**
- 添加 `mcm-code` ✅
- 添加 `mcm-data` ✅

### 步骤 3: 复制代码

打开 [`KAGGLE_ABLATION_NOTEBOOK.md`](./KAGGLE_ABLATION_NOTEBOOK.md)，复制所有 8 个 Cell 的代码到 Notebook。

### 步骤 4: 修改账号编号

**⚠️ 这是唯一需要修改的地方！**

在 **Cell 1** 中修改：

```python
# ============ 修改这里！ ============
ACCOUNT_ID = 1  # 👈 修改为 1-6
# ==================================
```

| Kaggle 账号 | ACCOUNT_ID | 任务 | 实验配置 |
|------------|------------|------|----------|
| Account 1  | `1`        | MATE | Baseline + Both |
| Account 2  | `2`        | MATE | CRF Only + Span Only |
| Account 3  | `3`        | MNER | Baseline + Both |
| Account 4  | `4`        | MNER | CRF Only + Span Only |
| Account 5  | `5`        | MABSA | Baseline + Both |
| Account 6  | `6`        | MABSA | CRF Only + Span Only |

### 步骤 5: 运行

点击 **"Run All"** 或逐个运行 Cell。

### 步骤 6: 下载结果

1. 等待所有 Cell 运行完成（约 5 小时）
2. 在 **Output** 标签下载 `account_X_results_YYYYMMDD_HHMMSS.zip`
3. 点击右上角 **"Stop Session"** 停止 Notebook（节省 GPU 配额）

### 步骤 7: 重复

在其他 5 个账号上重复步骤 1-6，记得修改 `ACCOUNT_ID`。

---

## 📊 所有账号完成后

### 1. 收集所有结果

将 6 个 zip 文件放到本地目录：

```
MCM/
  └── results/
      ├── account_1_results_20250103_120530.zip
      ├── account_2_results_20250103_121045.zip
      ├── account_3_results_20250103_122130.zip
      ├── account_4_results_20250103_123215.zip
      ├── account_5_results_20250103_124300.zip
      └── account_6_results_20250103_125345.zip
```

### 2. 解压所有文件

**Windows (PowerShell):**
```powershell
cd D:\Codes\MCL\MCM\results
Get-ChildItem account_*.zip | ForEach-Object {
    Expand-Archive -Path $_.FullName -DestinationPath ($_.BaseName) -Force
}
```

**Linux/Mac (Bash):**
```bash
cd /path/to/MCM/results
for zip in account_*.zip; do
    unzip -q "$zip" -d "${zip%.zip}"
done
```

### 3. 运行综合分析

```bash
cd D:\Codes\MCL\MCM
python scripts/configs/kaggle_ablation/analyze_results.py --results_dir ./results
```

**输出:**
- 📊 `ablation_comparison.csv` - 对比表格
- 📈 `ablation_visualization.png` - 可视化图表
- 📝 `ablation_report.md` - 详细分析报告

---

## ⏱️ 时间规划

| 阶段 | 时间 |
|------|------|
| 单个实验 | ~2.5 小时 |
| 单个账号（2个实验） | ~5 小时 |
| 6个账号（并行） | ~5 小时 |
| 6个账号（串行） | ~30 小时 |

**建议**: 6 个账号同时开跑，5 小时内全部完成 ✅

---

## ❓ 常见问题

### Q1: Cell 运行失败怎么办？

**A**: 检查以下项：
1. GPU 是否已启用
2. `mcm-code` 和 `mcm-data` 是否已添加
3. Internet 是否已开启
4. `ACCOUNT_ID` 是否正确（1-6）

### Q2: 如何确认实验正在运行？

**A**: 查看 Cell 7 的输出：
- 看到 `🚀 运行命令:` 表示已开始
- 看到训练日志（epoch 进度）表示正常运行
- 看到 `✅ XX 实验完成` 表示单个实验完成

### Q3: 实验完成后如何确认结果正确？

**A**: 检查 Cell 8 的输出：
- `总文件数` 应该 > 0
- `train_info` 应该 = 实验数量（通常是 2）
- zip 文件大小应该 > 0 MB

### Q4: 如果超过 12 小时限制怎么办？

**A**: 
- 当前配置：每个账号约 5 小时，远低于 12 小时限制
- 如果真的超时，可以手动减小 `num_epochs` 或 `batch_size`
- 或者分两次运行（每次 1 个实验）

### Q5: 如何查看进度？

**A**: 查看 `/kaggle/working/ablation_progress.json`：

```python
# 在 Notebook 中添加一个新 Cell
import json
with open("/kaggle/working/ablation_progress.json") as f:
    progress = json.load(f)
    print(json.dumps(progress, indent=2))
```

---

## 📋 检查清单

### 开始前
- [ ] 已生成本地配置（运行 `python scripts/generate_kaggle_ablation_configs.py`）
- [ ] 已打包并上传 `mcm-code.zip` 和 `mcm-data.zip` 到 Kaggle Datasets
- [ ] 准备好 6 个 Kaggle 账号

### 每个账号
- [ ] 创建新 Notebook
- [ ] 添加 GPU、Internet、Datasets
- [ ] 复制 8 个 Cell 代码
- [ ] **修改 Cell 1 中的 `ACCOUNT_ID`**
- [ ] 运行 "Run All"
- [ ] 等待完成（~5 小时）
- [ ] 下载结果 zip
- [ ] 停止 Session

### 所有账号完成后
- [ ] 收集 6 个 zip 文件
- [ ] 解压到本地
- [ ] 运行 `analyze_results.py`
- [ ] 查看分析报告

---

## 🎯 预期结果

消融实验将对比 4 种配置：

| 配置 | CRF | Span Loss | 预期 Chunk F1 |
|------|-----|-----------|---------------|
| Baseline | ❌ | ❌ | ~30% (基线) |
| CRF Only | ✅ | ❌ | ~50-60% (+20-30%) |
| Span Only | ❌ | ✅ | ~45-55% (+15-25%) |
| Both | ✅ | ✅ | **~60-75% (+30-45%)** ⭐ |

在 3 个任务上（MATE、MNER、MABSA）都应该看到类似的改进趋势。

---

**准备好了吗？开始你的消融实验！** 🚀

如有问题，请参考：
- 📖 详细指南: [`KAGGLE_ABLATION_NOTEBOOK.md`](./KAGGLE_ABLATION_NOTEBOOK.md)
- 📋 部署指南: [`KAGGLE_DEPLOYMENT_GUIDE.md`](./KAGGLE_DEPLOYMENT_GUIDE.md)
- 📚 项目文档: [`README.md`](./README.md)

