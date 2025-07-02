# 快速使用指南

## 🚀 一键生成所有脚本

```bash
# 生成所有标准化脚本
python scripts/generate_all_scripts.py
```

## 📋 常用脚本示例

### 服务器版本 (AutoDL)

```bash
# 多任务训练 - 无持续学习
./scripts/strain_AllTask_twitter2015_none_multi.sh

# 多任务训练 - EWC策略
./scripts/strain_AllTask_twitter2015_ewc_multi.sh

# 多任务训练 - Experience Replay
./scripts/strain_AllTask_twitter2015_replay_multi.sh

# 多任务训练 - 标签嵌入 + EWC
./scripts/strain_AllTask_twitter2015_ewc_multi.sh  # 自动包含标签嵌入
```

### 本地版本 (开发调试)

```bash
# 简化数据集 - 快速测试
./scripts/train_AllTask_200_none_multi.sh

# 简化数据集 - EWC测试
./scripts/train_AllTask_200_ewc_multi.sh

# 完整数据集 - 本地训练
./scripts/train_AllTask_twitter2015_none_multi.sh

# 单任务测试 - MABSA
./scripts/train_SingleTask_mabsa_200_none_multi.sh

# 单任务测试 - MNER (使用单独数据集)
./scripts/train_SingleTask_mner_200_none_multi.sh
```

## 🔧 自定义脚本生成

### 生成单个脚本

```bash
# 服务器版本 - 多任务 - Twitter2015 - EWC - 多模态
python scripts/config_templates.py \
    --env server \
    --task_type AllTask \
    --dataset twitter2015 \
    --strategy ewc \
    --mode multi

# 本地版本 - 简化数据集 - 无策略
python scripts/config_templates.py \
    --env local \
    --task_type AllTask \
    --dataset 200 \
    --strategy none \
    --mode multi

# 带标签嵌入的脚本
python scripts/config_templates.py \
    --env server \
    --task_type AllTask \
    --dataset twitter2015 \
    --strategy ewc \
    --mode multi \
    --use_label_embedding
```

## 🧹 清理旧脚本

```bash
# 备份并删除旧脚本，生成新脚本
python scripts/cleanup_scripts.py
```

## 📊 脚本命名规则

```
[环境]_[任务]_[数据集]_[策略]_[模式].sh

环境:
- strain_  # 服务器版本 (AutoDL)
- train_   # 本地版本

任务:
- AllTask     # 多任务训练 (MABSA + MASC + MATE + MNER)
- SingleTask  # 单任务训练 (指定具体任务)

数据集:
- twitter2015  # Twitter2015完整数据集
- twitter2017  # Twitter2017完整数据集
- 200          # 简化数据集 (200样本)

策略:
- none      # 无持续学习
- ewc       # Elastic Weight Consolidation
- replay    # Experience Replay
- lwf       # Learning without Forgetting
- si        # Synaptic Intelligence
- mas       # Memory Aware Synapses
- gem       # Gradient Episodic Memory
- mymethod  # 自定义方法
- tamcl     # TAM-CL
- moe       # MoE Adapters

模式:
- multi     # 多模态 (文本+图像)
- text      # 仅文本 (暂未实现)

## 📁 数据集说明

### 任务-数据集映射
- **MABSA/MASC/MATE任务**: 共享同一个数据集
  - Twitter2015: `data/MASC/twitter2015/`
  - Twitter2017: `data/MASC/twitter2017/`
  - 简化数据集: `data/MASC/twitter2015/train_100_samples.txt`

- **MNER任务**: 使用单独的数据集
  - Twitter2015: `data/MNER/twitter2015/`
  - Twitter2017: `data/MNER/twitter2017/`
  - 简化数据集: `data/MNER/twitter2015/train_100_samples.txt`
```

## 🎯 推荐使用流程

### 1. 本地开发阶段

```bash
# 生成本地测试脚本
python scripts/generate_all_scripts.py

# 使用简化数据集快速测试
./scripts/train_AllTask_200_none_multi.sh
./scripts/train_AllTask_200_ewc_multi.sh
```

### 2. 服务器训练阶段

```bash
# 上传代码到服务器后，使用服务器脚本
./scripts/strain_AllTask_twitter2015_ewc_multi.sh
./scripts/strain_AllTask_twitter2015_replay_multi.sh
```

### 3. 标签嵌入实验

```bash
# 生成带标签嵌入的脚本
python scripts/config_templates.py \
    --env server \
    --task_type AllTask \
    --dataset twitter2015 \
    --strategy ewc \
    --mode multi \
    --use_label_embedding

# 运行标签嵌入实验
./scripts/strain_AllTask_twitter2015_ewc_multi.sh
```

## 📁 文件存储位置

### 服务器版本

- 模型文件: `/root/autodl-tmp/checkpoints/1.pt`
- 日志文件: `/root/autodl-tmp/log/`
- EWC 参数: `/root/autodl-tmp/ewc_params/`
- 标签嵌入: `/root/autodl-tmp/checkpoints/label_embedding_*.pt`

### 本地版本

- 模型文件: `./checkpoints/{task}_{dataset}_{strategy}.pt`
- 日志文件: `./log/`
- EWC 参数: `./ewc_params/`
- 标签嵌入: `./checkpoints/label_embedding_*.pt`

## 🔍 查看训练结果

```bash
# 查看训练日志
tail -f ./log/mabsa_twitter2015_ewc.log

# 查看训练信息
cat ./checkpoints/train_info_mabsa_twitter2015_ewc.json

# 查看模型文件
ls -la ./checkpoints/
```

## ⚠️ 注意事项

1. **环境差异**: 服务器版本和本地版本使用不同的路径和命名规则
2. **数据集选择**: 本地开发建议使用 `200` 简化数据集，服务器使用完整数据集
3. **标签嵌入**: 新功能，建议在基础策略稳定后再使用
4. **脚本权限**: 生成的脚本会自动设置执行权限
5. **备份重要**: 清理旧脚本前会自动备份到 `scripts/backup_old_scripts/`

## 🆘 常见问题

### Q: 如何恢复旧脚本？

```bash
cp scripts/backup_old_scripts/旧脚本名.sh scripts/
chmod +x scripts/旧脚本名.sh
```

### Q: 如何修改脚本参数？

编辑生成的脚本文件，或使用 `config_templates.py` 重新生成

### Q: 如何添加新的持续学习策略？

修改 `scripts/config_templates.py` 中的 `strategies` 配置

### Q: 脚本运行失败怎么办？

1. 检查 CUDA 环境: `nvidia-smi`
2. 检查 Python 环境: `python --version`
3. 检查依赖: `pip list | grep torch`
4. 查看错误日志: `tail -f ./log/*.log`
