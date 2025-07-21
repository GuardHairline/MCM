# 训练脚本使用说明

## 概述

本项目提供了四个主要的 shell 脚本，用于自动化生成配置文件和运行训练：

1. `generate_all_local_configs.sh` - 生成本地版本的所有配置文件
2. `generate_all_server_configs.sh` - 生成服务器版本的所有配置文件
3. `run_all_local_training.sh` - 运行所有本地配置的训练
4. `run_all_server_training.sh` - 运行所有服务器配置的训练

## 脚本功能

### 配置文件生成脚本

#### `generate_all_local_configs.sh`

- **环境**: local
- **数据集**: 200 (简化数据集)
- **策略**: 所有 10 种持续学习策略
- **label_embedding**: 每个策略都有使用和不使用两个版本
- **生成文件**: 20 个配置文件 (10 策略 × 2 版本)

#### `generate_all_server_configs.sh`

- **环境**: server
- **数据集**: twitter2015, twitter2017, mix (3 个数据集)
- **策略**: 所有 10 种持续学习策略
- **label_embedding**: 每个策略都有使用和不使用两个版本
- **生成文件**: 60 个配置文件 (3 数据集 × 10 策略 × 2 版本)

### 训练运行脚本

#### `run_all_local_training.sh`

- 自动运行所有 `local_*.json` 配置文件
- 每个配置最多运行 2 小时
- 日志保存在 `scripts/log/local_training/`

#### `run_all_server_training.sh`

- 自动运行所有 `server_*.json` 配置文件
- 按数据集分组运行
- 每个配置最多运行 4 小时
- 日志保存在 `scripts/log/server_training/`

## 使用方法

### 1. 生成配置文件

```bash
# 生成本地版本配置文件
bash scripts/generate_all_local_configs.sh

# 生成服务器版本配置文件
bash scripts/generate_all_server_configs.sh
```

### 2. 运行训练

```bash
# 运行所有本地配置训练
bash scripts/run_all_local_training.sh

# 运行所有服务器配置训练
bash scripts/run_all_server_training.sh
```

## 支持的策略

1. **none** - 无持续学习策略
2. **ewc** - Elastic Weight Consolidation
3. **replay** - Experience Replay
4. **lwf** - Learning without Forgetting
5. **si** - Synaptic Intelligence
6. **mas** - Memory Aware Synapses
7. **gem** - Gradient Episodic Memory
8. **mymethod** - 自定义方法
9. **tam_cl** - TAM-CL
10. **moe** - MoE Adapters

## 任务序列

所有配置使用相同的任务序列：

```
masc → mate → mner → mabsa → masc → mate → mner → mabsa
```

模式序列：

```
text_only → text_only → text_only → text_only → multimodal → multimodal → multimodal → multimodal
```

## 文件命名规则

### 本地版本

- 格式: `local_200_[策略名](_label_emb).json`
- 示例: `local_200_none.json`, `local_200_ewc_label_emb.json`

### 服务器版本

- 格式: `server_[数据集]_[策略名](_label_emb).json`
- 示例: `server_twitter2015_none.json`, `server_mix_ewc_label_emb.json`

## 日志管理

- 本地训练日志: `scripts/log/local_training/`
- 服务器训练日志: `scripts/log/server_training/`
- 每个配置文件对应一个 `.log` 文件

## 注意事项

1. **时间限制**: 本地版本 2 小时，服务器版本 4 小时
2. **资源监控**: 服务器版本会显示系统资源使用情况
3. **错误处理**: 失败的训练会显示错误信息
4. **进度跟踪**: 显示完成统计和失败列表

## 手动运行单个配置

```bash
# 运行单个配置文件
python -m scripts.train_with_zero_shot --config scripts/configs/local_200_none.json

# 查看日志
tail -f scripts/log/local_training/local_200_none.log
```

## 故障排除

1. **配置文件不存在**: 先运行生成脚本
2. **训练超时**: 检查日志文件中的错误信息
3. **内存不足**: 减少 batch_size 或使用更小的数据集
4. **GPU 错误**: 检查 CUDA 版本和 GPU 内存

## 结果分析

训练完成后，结果保存在：

- 模型文件: `checkpoints/`
- 训练信息: `checkpoints/train_info_*.json`
- 持续学习指标: 包含在训练信息中
