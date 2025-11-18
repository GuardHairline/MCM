# AutoDL实验配置

## 🚀 快速开始

### 本地（Windows）

```bash
# 1. 生成配置
python scripts/generate_autodl_configs.py
```

### AutoDL服务器（Ubuntu）

```bash
# 2. 测试环境
bash scripts/configs/autodl_config/test_single_autodl.sh

# 3. 运行所有实验（后台）
nohup bash scripts/configs/autodl_config/run_autodl_experiments.sh > run.log 2>&1 &

# 4. 查看进度
bash scripts/configs/autodl_config/check_progress.sh

# 5. 监控日志
tail -f checkpoints/autodl/log/autodl_run_*.log
```

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `run_autodl_experiments.sh` | 🔥 主运行脚本（自动关机） |
| `test_single_autodl.sh` | 测试单个配置 |
| `check_progress.sh` | 查看进度 |
| `stop_experiments.sh` | 紧急停止 |
| `cancel_shutdown.sh` | 取消关机 |
| `config_index.json` | 配置索引 |
| `autodl_*.json` | 实验配置（共54个） |

---

## ⚠️ 重要提示

### 关机机制

- ✅ 所有实验完成后**自动关机**
- ⏱️ 10秒倒计时
- 🛑 Ctrl+C 或运行 `cancel_shutdown.sh` 取消

### 费用提醒

- 💰 AutoDL按时计费
- 📊 预计运行时间：25-30小时
- 💾 记得定期下载checkpoint

---

## 🔧 常用命令

```bash
# 查看运行状态
ps aux | grep train_with_zero_shot

# 查看GPU
nvidia-smi

# 查看磁盘
df -h

# 查看进程日志
tail -f checkpoints/autodl/log/*.log

# 停止实验
bash scripts/configs/autodl_config/stop_experiments.sh
```

---

## 📚 完整文档

详细使用指南请查看：`AUTODL_SETUP_GUIDE.md`







