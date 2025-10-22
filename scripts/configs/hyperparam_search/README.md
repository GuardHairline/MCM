# MASC 超参数搜索实验

## 概述

本目录包含 **52** 个超参数搜索实验配置，用于MASC任务序列的最优超参数搜索。

**任务序列**: masc(text_only) -> masc(multimodal)  
**测试策略**: none, replay, moe, deqa  
**超参数**: lr, step_size, gamma

## 文件说明

- `config_index.json` - 所有配置文件的索引
- `*.json` - 各个实验的配置文件
- `run_all_experiments.sh` - 主实验运行脚本（需要SSH保持连接）
- `start_experiments_detached.sh` - **推荐**：后台启动脚本（可脱离SSH）
- `stop_all_experiments.sh` - 停止所有实验
- `logs/` - 实验日志目录

## 使用方法

### ⚠️ 重要提示

脚本会自动切换到项目根目录运行，确保：
1. 脚本从正确的位置执行
2. 能够正确导入 `scripts.train_with_zero_shot` 模块
3. 日志输出会显示当前工作目录，请检查是否正确

### 生成配置时的选项

如果需要从特定实验开始（例如已完成前2个实验），在生成配置时使用：

```bash
# 从第3个实验开始生成脚本
python -m scripts.generate_masc_hyperparameter_configs --start_exp_id 3

# 这样生成的脚本会从exp_3开始编号，避免覆盖已有结果
```

### 方法1：后台运行（推荐，可脱离SSH）

```bash
# 启动所有实验（可以断开SSH）
bash scripts/configs/hyperparam_search/start_experiments_detached.sh

# 断开SSH连接也没问题，实验会继续运行

# 检查主日志，确认项目根目录是否正确
tail -f scripts/configs/hyperparam_search/logs/master_*.log | head -20
```

### 方法2：使用 tmux（推荐）

```bash
# 创建tmux会话
tmux new -s masc_hyperparam

# 运行实验
bash scripts/configs/hyperparam_search/run_all_experiments.sh

# 断开会话（实验继续运行）：Ctrl+B 然后按 D
# 重新连接：tmux attach -t masc_hyperparam
```

### 方法3：使用 screen

```bash
# 创建screen会话
screen -S masc_hyperparam

# 运行实验
bash scripts/configs/hyperparam_search/run_all_experiments.sh

# 断开会话：Ctrl+A 然后按 D
# 重新连接：screen -r masc_hyperparam
```

## 监控实验

### 查看主日志
```bash
tail -f scripts/configs/hyperparam_search/logs/master_*.log
```

### 查看单个实验日志
```bash
ls -lth scripts/configs/hyperparam_search/logs/exp_*.log
tail -f scripts/configs/hyperparam_search/logs/exp_1_none_*.log
```

### 查看所有运行中的进程
```bash
ps aux | grep train_with_zero_shot
```

### 监控GPU使用
```bash
watch -n 1 nvidia-smi
```

### 查看实验进度
```bash
# 查看有多少实验已完成
grep "completed successfully" scripts/configs/hyperparam_search/logs/*.log | wc -l

# 查看有多少实验正在运行
ps aux | grep train_with_zero_shot | grep -v grep | wc -l
```

## 停止实验

### 停止所有实验
```bash
bash scripts/configs/hyperparam_search/stop_all_experiments.sh
```

### 停止单个实验
```bash
# 查找PID
ps aux | grep train_with_zero_shot

# 停止特定进程
kill <PID>
```

## GPU分配策略

脚本会自动检测可用GPU并智能分配：

- **2张GPU**: 实验会轮流分配到GPU0和GPU1，充分利用两张卡
- **多张GPU**: 自动并行利用所有可用GPU
- **单张GPU**: 实验会串行执行，前一个完成后启动下一个

### GPU空闲判断标准

- **空闲标准**: GPU使用率 < 10%
- **超时后保护**: 如果等待超时，会检查GPU使用率和显存使用率
  - 如果使用率 > 50% 或 显存使用 > 50%，继续等待
  - 避免干扰其他正在运行的程序

## 实验配置

总共 **52** 个配置，包括：

- 4种策略 (none, replay, moe, deqa)
- 多组超参数组合 (lr, step_size, gamma)
- 每个实验固定20个epoch
- **早停已禁用** (patience=999)，确保所有实验完整训练以公平比较

详细配置请查看 `config_index.json`

## 结果分析

实验完成后，结果会保存在：

- 模型文件: `checkpoints/twitter2015_<strategy>_<mode>_<seq>.pt`
- 训练信息: `checkpoints/train_info_twitter2015_<strategy>_<mode>_<seq>.json`
- 准确率热力图: `checkpoints/acc_matrix/`

## 故障排查

### 问题：No module named scripts.train_with_zero_shot
```bash
# 这通常是因为工作目录不正确
# 检查主日志中的"项目根目录"和"当前工作目录"
tail -n 50 scripts/configs/hyperparam_search/logs/master_*.log | grep "目录"

# 应该看到类似输出：
# 项目根目录: /path/to/MCM
# 当前工作目录: /path/to/MCM

# 如果路径不对，手动运行脚本时请从项目根目录执行
```

### 问题：进程启动失败
```bash
# 检查主日志
tail -n 50 scripts/configs/hyperparam_search/logs/master_*.log

# 检查实验日志
tail -n 50 scripts/configs/hyperparam_search/logs/exp_1_*.log
```

### 问题：GPU显存不足
```bash
# 查看GPU使用情况
nvidia-smi

# 可能需要减少并行实验数量，编辑 run_all_experiments.sh
# 在 wait_for_free_gpu 函数中调整显存阈值
```

### 问题：SSH断开后进程停止
```bash
# 确保使用了后台启动脚本
bash scripts/configs/hyperparam_search/start_experiments_detached.sh

# 或者使用tmux/screen
```

## 注意事项

1. **确保有足够的磁盘空间**：每个实验会生成模型文件和日志
2. **监控GPU温度**：长时间运行可能导致GPU过热
3. **定期检查日志**：及时发现和处理错误
4. **实验命名**：配置文件名已经包含了超参数信息，便于识别

## 联系方式

如有问题，请查看主项目README或联系开发者。
