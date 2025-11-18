# Kaggle BiLSTM测试部署指南

## 实验目的

测试新实现的BiLSTM-CRF任务头在MATE、MNER、MABSA三个任务上的表现，并比较不同超参数配置的效果。

## 实验设计

### 任务分配

| 账号 | 任务 | 配置 | 预计时间 |
|------|------|------|----------|
| Account 1 | MATE | small, default, large | ~9小时 |
| Account 2 | MNER | small, default, large | ~10小时 |
| Account 3 | MABSA | small, default, large | ~11小时 |

### BiLSTM配置

1. **config_small**: 
   - hidden_size=128, num_layers=1
   - 快速训练，基线对比

2. **config_default**: 
   - hidden_size=256, num_layers=2
   - 推荐的默认配置

3. **config_large**: 
   - hidden_size=512, num_layers=2
   - 更大容量，可能效果更好但训练慢

### 持续学习序列

每个配置包含2个session的持续学习序列：
1. **text_only**: 仅使用文本模态
2. **multimodal**: 使用文本+图像（继承text_only的权重）

## 部署步骤

### 1. 准备Kaggle数据集

确保已上传 `mcm-project` 数据集，包含：
- 完整项目代码
- 数据文件 (Twitter2015)
- 预训练模型

### 2. 为每个账号创建Notebook

对每个账号 (account_1, account_2, account_3)：

1. 登录对应的Kaggle账号
2. 创建新Notebook
3. 设置：
   - Accelerator: GPU T4 或 P100
   - Internet: On (如需下载预训练模型)
4. 添加数据集: `mcm-project`
5. 复制对应的 `run_account_X.py` 内容到Notebook
6. 点击 "Run All"

### 3. 监控执行

- 每个账号运行约9-11小时
- 关注训练日志，确认没有错误
- 检查GPU使用率

### 4. 导出结果

完成后，从 `/kaggle/working/results_account_X/` 下载：
- `train_info_*.json`: 训练信息
- `*.png`: 热力图
- `run_summary_*.json`: 运行摘要

## 预期结果

### 关键指标

- **Chunk F1 (Span-level)**: 序列任务的主要评估指标
- **Token Micro F1**: token级别的F1分数
- **Training Time**: 训练时间对比

### 对比维度

1. **任务间对比**: MATE vs MNER vs MABSA
2. **配置间对比**: small vs default vs large
3. **模态对比**: text_only vs multimodal
4. **持续学习效果**: 从text_only到multimodal的迁移

## 结果分析

运行 `analyze_bilstm_results.py` 自动生成：
- 性能对比表格
- 训练时间分析
- 超参数影响分析
- 最佳配置推荐

## 故障排除

### 常见问题

1. **OOM (Out of Memory)**
   - 减小 batch_size
   - 使用 config_small

2. **训练时间过长**
   - 减少 epochs
   - 只运行部分配置

3. **数据集未找到**
   - 确认 mcm-project 数据集已添加
   - 检查数据集版本

## 下一步

1. 收集所有账号的结果
2. 运行分析脚本
3. 根据结果选择最佳配置
4. 在完整数据集上进行完整实验
