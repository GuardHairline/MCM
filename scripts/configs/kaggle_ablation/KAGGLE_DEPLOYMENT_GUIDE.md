# Kaggle 6账号Ablation Study部署指南

## 📋 实验设计

### 总体方案

- **总配置数**: 12个 (3任务 × 4 ablation)
- **账号数**: 6个
- **每账号配置数**: 2个
- **每账号预计时间**: 3-4小时
- **Kaggle限制**: 12小时（充足余量）

### 账号分配

| 账号 | 任务 | Ablation配置 | 预计时间 |
|------|------|--------------|----------|
| **Account 1** | MATE | baseline + crf_and_span | ~3小时 |
| **Account 2** | MATE | crf_only + span_only | ~3小时 |
| **Account 3** | MNER | baseline + crf_and_span | ~3.3小时 |
| **Account 4** | MNER | crf_only + span_only | ~3.3小时 |
| **Account 5** | MABSA | baseline + crf_and_span | ~3.6小时 |
| **Account 6** | MABSA | crf_only + span_only | ~3.6小时 |

## 🚀 部署步骤

### 1. 准备Kaggle数据集

在本地执行以下步骤：

```bash
# 1. 打包项目
cd /path/to/MCM
zip -r mcm-project.zip . -x "*.git*" "*__pycache__*" "*.pyc" "*checkpoint*" "*logs*"

# 2. 上传到Kaggle
# 在Kaggle网站上:
# - 点击 "Datasets" -> "New Dataset"
# - 上传 mcm-project.zip
# - 设置名称为 "mcm-project"
# - 设置为 Private
# - 点击 "Create"
```

### 2. 为每个账号创建Notebook

对于每个账号（1-6），重复以下步骤：

#### Step 1: 创建Notebook

1. 登录对应的Kaggle账号
2. 点击 "Code" -> "New Notebook"
3. Notebook设置：
   - **加速器**: GPU P100
   - **Internet**: On
   - **持久化**: Off (节省配额)

#### Step 2: 添加数据集

1. 点击右侧 "Add Data"
2. 搜索 "mcm-project"
3. 添加你的数据集

#### Step 3: 复制运行脚本

1. 打开对应账号的运行脚本:
   - Account 1: `account_1/run_account_1.py`
   - Account 2: `account_2/run_account_2.py`
   - ... (以此类推)

2. 复制全部内容到Notebook

3. 修改数据集名称（如果需要）:
   ```python
   PROJECT_DATASET = "your-username/mcm-project"  # 修改为你的数据集路径
   ```

#### Step 4: 运行

1. 点击 "Save Version"
2. 选择 "Save & Run All (Commit)"
3. 等待完成（3-4小时）

### 3. 同时运行所有账号

✨ **关键**: 6个账号可以同时运行，互不干扰！

```
Account 1  →  [MATE baseline + full]     →  3小时  →  完成
Account 2  →  [MATE crf + span]          →  3小时  →  完成
Account 3  →  [MNER baseline + full]     →  3.3小时 →  完成
Account 4  →  [MNER crf + span]          →  3.3小时 →  完成
Account 5  →  [MABSA baseline + full]    →  3.6小时 →  完成
Account 6  →  [MABSA crf + span]         →  3.6小时 →  完成

总时间: ~3.6小时 (并行)
```

## 📊 收集结果

### 每个账号完成后

1. 进入对应的Notebook
2. 点击 "Output"
3. 下载文件:
   - `account_X_final_results.json` (必需)
   - `checkpoints/` 目录下的模型文件 (可选)

### 文件组织

```
results/
├── account_1_final_results.json
├── account_2_final_results.json
├── account_3_final_results.json
├── account_4_final_results.json
├── account_5_final_results.json
└── account_6_final_results.json
```

## 🔍 结果分析

下载所有结果后，运行分析脚本：

```bash
cd scripts/configs/kaggle_ablation

# 将下载的JSON文件放到results/目录

python analyze_results.py
```

这将生成：
- `ablation_study_summary.json` - 总结
- `ablation_study_report.md` - 详细报告
- `ablation_comparison.png` - 对比图表

## ⚠️ 注意事项

### Kaggle限制

- **GPU时间**: 每周30小时（每账号）
- **运行时间**: 单次最多12小时
- **并行**: 每账号最多1个active session

### 最佳实践

1. **监控进度**: 定期检查Output日志
2. **及时保存**: 实验完成后立即下载结果
3. **备份**: 保存所有JSON文件
4. **网络稳定**: 确保上传数据集时网络稳定

### 故障处理

**问题1: 数据集未找到**
```
❌ 数据集未找到: /kaggle/input/mcm-project
```
解决: 检查数据集是否正确添加，名称是否匹配

**问题2: GPU不可用**
```
❌ CUDA available: False
```
解决: 检查Notebook设置，确保选择了GPU P100加速器

**问题3: 时间超限**
```
Session timeout after 12 hours
```
解决: 减少epochs或batch_size（但我们的配置应该不会超时）

## 📈 预期结果

### Chunk F1提升

| 配置 | MATE | MNER | MABSA |
|------|------|------|-------|
| **Baseline** | ~32% | ~30% | ~35% |
| **CRF only** | ~68% (+36%) | ~65% (+35%) | ~70% (+35%) |
| **Span only** | ~65% (+33%) | ~62% (+32%) | ~67% (+32%) |
| **CRF + Span** | ~76% (+44%) | ~74% (+44%) | ~78% (+43%) |

## 🎉 完成检查清单

- [ ] 6个账号的Notebook都已创建
- [ ] 所有Notebook都添加了mcm-project数据集
- [ ] 所有Notebook都设置了GPU P100
- [ ] 所有运行脚本已正确复制
- [ ] 6个Notebook同时运行
- [ ] 所有结果文件已下载
- [ ] 结果分析脚本已运行
- [ ] 分析报告已生成

---

**准备好了吗？开始你的Ablation Study之旅！** 🚀
