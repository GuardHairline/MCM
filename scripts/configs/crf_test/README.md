# CRF修复测试配置

## 📝 概述

这是一个用于验证CRF、valid_len修复和Span Loss效果的测试配置集。

**目的**：快速验证新代码修复在三个序列标注任务上的表现
**任务**：MATE、MNER、MABSA（各1个实验，共3个）
**预期**：Chunk F1从30%提升到60-75%

---

## 🎯 三大修复

### 1. CRF层（+15-30%）
- **问题**：无BIO约束，预测非法标签序列
- **解决**：添加CRF层，Viterbi解码
- **文件**：`models/task_heads/*_head.py`

### 2. valid_len修复（+5-10%）
- **问题**：计算valid_len时多+1，包含padding
- **解决**：移除+1，精确切片
- **文件**：`modules/evaluate.py`

### 3. Span Loss（+5-15%）
- **问题**：只优化token级别，不关注边界
- **解决**：添加边界loss，强化B标签权重
- **文件**：`utils/span_loss.py`, `modules/training_loop_fixed.py`

---

## 🚀 快速开始

### 本地运行

```bash
# 1. 生成配置
python scripts/generate_crf_test_configs.py

# 2. 运行测试
./scripts/configs/crf_test/run_crf_tests.sh

# 或单独运行
python scripts/train_with_zero_shot.py \
  --config scripts/configs/crf_test/crf_test_twitter2015_mate.json
```

### Kaggle运行

```bash
# 1. 生成Kaggle配置
python scripts/generate_crf_test_configs.py --kaggle

# 2. 按照KAGGLE_CRF_TEST_GUIDE.md部署
# 3. 在Kaggle Notebook中运行
```

---

## 📁 生成的文件

```
scripts/configs/crf_test/
├── crf_test_twitter2015_mate.json    # MATE配置
├── crf_test_twitter2015_mner.json    # MNER配置
├── crf_test_twitter2015_mabsa.json   # MABSA配置
├── test_index.json                    # 索引文件
├── run_crf_tests.sh                   # 批量运行脚本
├── README.md                          # 本文档
└── KAGGLE_CRF_TEST_GUIDE.md          # Kaggle完整指南
```

---

## 🔍 验证修复

### 检查CRF

训练日志应显示：
```
[MATE] Head initialized with CRF (num_labels=3)
✓ Span Loss enabled for mate (boundary_weight=0.2)
```

### 检查指标

评估结果应显示：
```
[mate] Chunk F1: 65.23% (主指标1) ← 关注这个！
Token Micro F1 (无O): 88.45% (主指标2)
Token Acc: 90.12% (参考)
```

### 性能对比

| 指标 | 修复前 | 修复后 | 提升 |
|------|--------|--------|------|
| Token microF1 | 90% | 90-92% | +0-2% |
| **Chunk F1** | **30%** | **60-75%** | **+30-45%** ⭐ |
| Boundary Precision | 40% | 70-80% | +30-40% |
| Boundary Recall | 35% | 65-75% | +30-40% |

---

## 📊 配置详情

### 超参数（推荐值）

```json
{
  "lr": 1e-5,
  "step_size": 10,
  "gamma": 0.5,
  "epochs": 20,
  "patience": 5,
  "batch_size": 16
}
```

### CRF配置

```json
{
  "use_crf": true,
  "use_span_loss": true,
  "boundary_weight": 0.2,
  "span_f1_weight": 0.0,
  "transition_weight": 0.0
}
```

### 任务标签数

- MATE: 3 (O, B, I)
- MNER: 9 (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC)
- MABSA: 7 (O, B-NEG, I-NEG, B-NEU, I-NEU, B-POS, I-POS)

---

## 📈 预期结果

### MATE任务

```
修复前：microF1=90%, chunkF1=30%
修复后：microF1=90%, chunkF1=65%
提升：+35%
```

### MNER任务

```
修复前：microF1=89%, chunkF1=28%
修复后：microF1=90%, chunkF1=72%
提升：+44%
```

### MABSA任务

```
修复前：microF1=91%, chunkF1=32%
修复后：microF1=91%, chunkF1=69%
提升：+37%
```

---

## 🛠️ 定制配置

### 修改数据集

```bash
python scripts/generate_crf_test_configs.py \
  --dataset twitter2017
```

### 修改超参数

编辑 `scripts/generate_crf_test_configs.py`:

```python
self.recommended_hyperparams = {
    "lr": 5e-6,           # 降低学习率
    "boundary_weight": 0.3  # 增加边界权重
}
```

### 禁用某个修复

编辑配置文件：

```json
{
  "use_crf": false,        # 禁用CRF
  "use_span_loss": false   # 禁用Span Loss
}
```

---

## 📚 相关文档

### 完整指南
- 📖 **Kaggle部署**: `KAGGLE_CRF_TEST_GUIDE.md`
- 📖 **修复详情**: `../../../doc/FIXES_GUIDE.md`
- 📖 **快速参考**: `../../../doc/FIXES_SUMMARY.md`

### 测试验证
- 🧪 **torchcrf检查**: `tests/test_torchcrf_availability.py`
- 🧪 **valid_len测试**: `tests/test_valid_len_fix.py`
- 🧪 **综合验证**: `tests/test_fixes_validation.py`

### 代码文件
- 🔧 **CRF实现**: `models/task_heads/token_label_heads.py`
- 🔧 **任务头**: `models/task_heads/{mate,mner,mabsa}_head.py`
- 🔧 **Span Loss**: `utils/span_loss.py`
- 🔧 **训练循环**: `modules/training_loop_fixed.py`

---

## ⚠️ 注意事项

### 1. 需要重新训练

旧checkpoint不包含CRF参数，无法直接加载。

### 2. torchcrf可选

如果没有安装 `pytorch-crf`，会自动使用内置 `SimpleCRF`。

安装方法（可选）：
```bash
pip install pytorch-crf
```

### 3. 主指标变更

- 旧指标：token-level microF1
- **新指标：chunk-level F1** ⭐

### 4. 时间成本

- CRF增加20-30%训练时间
- 但chunk F1提升30-45%，非常值得

---

## 🔍 故障排查

### 问题1：CRF未启用

**日志中未看到**：
```
[MATE] Head initialized with CRF
```

**解决**：
1. 检查配置文件中 `use_crf: true`
2. 重新生成配置
3. 确认代码是最新版本

### 问题2：Chunk F1未提升

**可能原因**：
1. CRF未正确启用
2. Span Loss未启用
3. 训练轮数不足
4. 数据集问题

**解决**：
1. 查看训练日志确认修复已启用
2. 增加训练轮数（epochs=30）
3. 检查数据集完整性

### 问题3：CUDA out of memory

**解决**：
```json
{
  "batch_size": 8  // 减小batch size
}
```

---

## 💡 使用建议

### 1. 快速验证流程

```bash
# 第1步：生成配置
python scripts/generate_crf_test_configs.py

# 第2步：运行单个测试验证
python scripts/train_with_zero_shot.py \
  --config scripts/configs/crf_test/crf_test_twitter2015_mate.json

# 第3步：检查Chunk F1是否提升
cat checkpoints/train_info_*.json | grep chunk_f1

# 第4步：如果提升明显，运行全部测试
./scripts/configs/crf_test/run_crf_tests.sh
```

### 2. 对比实验

运行一个启用CRF和一个不启用CRF的实验：

```bash
# 启用CRF（默认）
python scripts/train_with_zero_shot.py \
  --config scripts/configs/crf_test/crf_test_twitter2015_mate.json

# 禁用CRF（手动修改配置）
# 编辑配置文件，设置 use_crf: false
python scripts/train_with_zero_shot.py \
  --config scripts/configs/crf_test/crf_test_twitter2015_mate_no_crf.json

# 对比两者的Chunk F1
```

### 3. Kaggle部署

对于GPU资源有限的情况，推荐使用Kaggle：

```bash
# 1. 生成Kaggle配置
python scripts/generate_crf_test_configs.py --kaggle

# 2. 按照KAGGLE_CRF_TEST_GUIDE.md完整部署

# 3. 在Kaggle上运行（免费GPU）

# 4. 下载结果分析
```

---

## 📞 获取帮助

### 查看详细指南

```bash
# 查看修复指南
cat doc/FIXES_GUIDE.md

# 查看快速参考
cat doc/FIXES_SUMMARY.md

# 查看Kaggle指南
cat scripts/configs/crf_test/KAGGLE_CRF_TEST_GUIDE.md
```

### 运行测试验证

```bash
# 验证torchcrf
python tests/test_torchcrf_availability.py

# 验证valid_len修复
python tests/test_valid_len_fix.py

# 综合验证
python tests/test_fixes_validation.py
```

---

## ✅ 成功标志

运行成功后，你应该看到：

1. ✅ 训练日志显示CRF已初始化
2. ✅ Span Loss已启用
3. ✅ Chunk F1提升30-45%
4. ✅ 边界Precision/Recall明显提升
5. ✅ Token microF1保持90%左右

---

## 🎉 下一步

测试成功后，可以：

1. 将CRF和Span Loss应用到所有序列任务
2. 进行超参数调优
3. 在其他数据集上验证
4. 结合持续学习策略（EWC、LwF等）

---

**祝测试顺利！** 🚀

如有问题，请参考 `doc/FIXES_GUIDE.md` 或运行测试脚本验证。

