# Kaggle 6账号 CRF & Span Loss 消融实验

## 🚀 快速开始（3步上手）

### 1️⃣ 生成配置（本地执行）

```bash
cd D:\Codes\MCL\MCM
python scripts/generate_kaggle_ablation_configs.py
```

### 2️⃣ 在 Kaggle 上运行（6个账号并行）

**📖 完整指南**: [`QUICK_START.md`](./QUICK_START.md) ⭐ **推荐先看这个！**

**快速步骤**:
1. 打开 [`KAGGLE_ABLATION_NOTEBOOK.md`](./KAGGLE_ABLATION_NOTEBOOK.md)
2. 复制所有 8 个 Cell 代码到 Kaggle Notebook
3. **仅修改 Cell 1 中的 `ACCOUNT_ID`**（1-6）
4. 点击 "Run All"
5. 下载结果 zip

### 3️⃣ 分析结果（本地执行）

```bash
# 将6个账号的结果zip下载到 results/ 目录后
cd D:\Codes\MCL\MCM
python scripts/configs/kaggle_ablation/analyze_results.py --results_dir ./results
```

---

## 📋 账号分配

| 账号 | ACCOUNT_ID | 任务 | 实验配置 | 预计时间 |
|------|-----------|------|----------|---------|
| Account 1 | `1` | MATE | Baseline + Both | ~5h |
| Account 2 | `2` | MATE | CRF Only + Span Only | ~5h |
| Account 3 | `3` | MNER | Baseline + Both | ~5h |
| Account 4 | `4` | MNER | CRF Only + Span Only | ~5h |
| Account 5 | `5` | MABSA | Baseline + Both | ~5h |
| Account 6 | `6` | MABSA | CRF Only + Span Only | ~5h |

**总时间**: 6个账号并行 ~5小时 ✅（远低于 Kaggle 12小时限制）

---

## 📚 完整文档

### 🌟 核心文档（按使用顺序）

1. **[QUICK_START.md](./QUICK_START.md)** ⭐ - 一分钟上手指南
2. **[KAGGLE_ABLATION_NOTEBOOK.md](./KAGGLE_ABLATION_NOTEBOOK.md)** - 完整 Notebook 代码（8个Cell）
3. **[KAGGLE_DEPLOYMENT_GUIDE.md](./KAGGLE_DEPLOYMENT_GUIDE.md)** - 详细部署指南

### 📖 参考文档

- **CRF 测试参考**: [`scripts/configs/crf_test/KAGGLE_CRF_GUIDE_SETUP.md`](../crf_test/KAGGLE_CRF_GUIDE_SETUP.md)
- **CRF 原理**: [`doc/LOSS_AND_CRF_EXPLAINED.md`](../../../doc/LOSS_AND_CRF_EXPLAINED.md)

---

## 🎯 实验目标

**对比 4 种配置在 3 个任务上的效果**:

| 配置 | CRF | Span Loss | 预期效果 |
|------|-----|-----------|----------|
| **Baseline** | ❌ | ❌ | Chunk F1 ~30% (基线) |
| **CRF Only** | ✅ | ❌ | Chunk F1 ~50-60% (+20-30%) |
| **Span Only** | ❌ | ✅ | Chunk F1 ~45-55% (+15-25%) |
| **Both** | ✅ | ✅ | **Chunk F1 ~60-75% (+30-45%)** ⭐ |

在 **MATE**、**MNER**、**MABSA** 三个任务上都应该看到类似改进。

---

## ⏱️ 时间估算

- **单个实验**: ~2.5 小时
- **单个账号**（2个实验）: ~5 小时
- **6个账号并行**: ~5 小时 ✅
- **Kaggle 限制**: 12 小时
- **安全余量**: 7 小时 ✅

---

## 📊 结果分析

所有账号完成后，使用 `analyze_results.py` 将生成：

1. **对比表格** (`ablation_comparison.csv`)
   - 每个任务在 4 种配置下的详细指标
   - Chunk F1, Token F1, Precision, Recall 等

2. **可视化图表** (`ablation_visualization.png`)
   - 柱状图对比各配置的 Chunk F1
   - 折线图显示改进趋势

3. **分析报告** (`ablation_report.md`)
   - 统计显著性检验
   - 最佳配置推荐
   - 结论和建议

---

## 📂 目录结构

```
kaggle_ablation/
├── README.md                          # 本文件
├── QUICK_START.md                     # 快速上手指南 ⭐
├── KAGGLE_ABLATION_NOTEBOOK.md        # Notebook 代码（8个Cell）
├── KAGGLE_DEPLOYMENT_GUIDE.md         # 详细部署指南
├── analyze_results.py                 # 结果分析脚本
├── master_index.json                  # 总索引
│
├── account_1/                         # Account 1 配置
│   ├── account_1_index.json
│   ├── kaggle_baseline_twitter2015_mate.json
│   ├── kaggle_crf_and_span_twitter2015_mate.json
│   └── run_account_1.py
│
├── account_2/                         # Account 2 配置
│   └── ...
│
├── ... (account_3 到 account_6)
│
└── results/                           # 结果目录（手动创建）
    ├── account_1_results_*.zip
    ├── account_2_results_*.zip
    └── ...
```

---

## ✅ 检查清单

### 本地准备
- [ ] 运行 `python scripts/generate_kaggle_ablation_configs.py`
- [ ] 打包并上传 `mcm-code.zip` 和 `mcm-data.zip` 到 Kaggle Datasets

### 每个 Kaggle 账号
- [ ] 创建新 Notebook
- [ ] 配置 GPU (P100/T4) + Internet
- [ ] 添加 `mcm-code` 和 `mcm-data` 数据集
- [ ] 复制 Notebook 代码（8个Cell）
- [ ] **修改 `ACCOUNT_ID` (1-6)**
- [ ] 运行 "Run All"
- [ ] 下载结果 zip
- [ ] 停止 Session（节省GPU配额）

### 结果分析
- [ ] 收集所有 6 个结果 zip 文件
- [ ] 解压到本地 `results/` 目录
- [ ] 运行 `python analyze_results.py --results_dir ./results`
- [ ] 查看生成的对比表格和图表

---

## ❓ 常见问题

### Q: 为什么要分 6 个账号？
**A**: 
- 6 个账号并行运行，总时间 ~5 小时
- 如果单个账号跑 12 个实验，需要 ~30 小时，超过 Kaggle 限制

### Q: 每个账号只需要修改什么？
**A**: 
- **仅修改 Cell 1 中的 `ACCOUNT_ID`**（1-6）
- 其他代码完全不需要修改

### Q: 如何确认实验正在运行？
**A**: 
- 看到训练日志（epoch 进度）表示正常运行
- 可以查看 `/kaggle/working/ablation_progress.json` 查看进度

### Q: 结果在哪里下载？
**A**: 
- 在 Kaggle Notebook 的 **Output** 标签
- 文件名: `account_X_results_YYYYMMDD_HHMMSS.zip`

### Q: 如何避免超时？
**A**: 
- 当前配置每个账号 ~5 小时，远低于 12 小时限制
- 如果担心，可以适当减小 `num_epochs` 或 `batch_size`

---

## 🔗 相关链接

- **项目主README**: [`../../README.md`](../../README.md)
- **修复详情**: [`doc/ALL_FIXES_SUMMARY.md`](../../../doc/ALL_FIXES_SUMMARY.md)
- **快速参考**: [`doc/QUICK_REFERENCE.md`](../../../doc/QUICK_REFERENCE.md)

---

**准备好了吗？从 [QUICK_START.md](./QUICK_START.md) 开始！** 🚀
