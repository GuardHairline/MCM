# 训练脚本索引

## 数据集说明
- **MABSA/MASC/MATE任务**: 共享同一个数据集 (MASC目录下的Twitter2015/Twitter2017)
- **MNER任务**: 使用单独的数据集 (MNER目录下的Twitter2015/Twitter2017)

## 服务器版本 (strain_*)
服务器版本的所有模型都保存为 `1.pt`，文件存储在 `/root/autodl-tmp/` 目录下。

### 多任务训练
- `strain_AllTask_twitter2015_none_multi.sh` - 无持续学习策略
- `strain_AllTask_twitter2015_ewc_multi.sh` - EWC策略
- `strain_AllTask_twitter2015_replay_multi.sh` - Experience Replay
- `strain_AllTask_twitter2015_lwf_multi.sh` - Learning without Forgetting
- `strain_AllTask_twitter2015_si_multi.sh` - Synaptic Intelligence
- `strain_AllTask_twitter2015_mas_multi.sh` - Memory Aware Synapses
- `strain_AllTask_twitter2015_mymethod_multi.sh` - 自定义方法
- `strain_AllTask_twitter2015_moe_multi.sh` - MoE Adapters

### Twitter2017数据集
- `strain_AllTask_twitter2017_none_multi.sh` - 无持续学习
- `strain_AllTask_twitter2017_ewc_multi.sh` - EWC策略
- `strain_AllTask_twitter2017_replay_multi.sh` - Experience Replay

### 单任务训练
- `strain_SingleTask_mabsa_twitter2015_none_multi.sh` - MABSA单任务
- `strain_SingleTask_masc_twitter2015_none_multi.sh` - MASC单任务
- `strain_SingleTask_mate_twitter2015_none_multi.sh` - MATE单任务
- `strain_SingleTask_mner_twitter2015_none_multi.sh` - MNER单任务

### 标签嵌入版本
- `strain_AllTask_twitter2015_none_multi.sh` - 无策略 + 标签嵌入
- `strain_AllTask_twitter2015_ewc_multi.sh` - EWC + 标签嵌入

## 本地版本 (train_*)
本地版本使用详细命名，文件存储在当前目录下。

### 简化数据集 (200样本)
- `train_AllTask_200_none_multi.sh` - 无持续学习
- `train_AllTask_200_ewc_multi.sh` - EWC策略
- `train_AllTask_200_replay_multi.sh` - Experience Replay
- `train_AllTask_200_lwf_multi.sh` - Learning without Forgetting
- `train_AllTask_200_si_multi.sh` - Synaptic Intelligence
- `train_AllTask_200_mas_multi.sh` - Memory Aware Synapses
- `train_AllTask_200_mymethod_multi.sh` - 自定义方法

### 完整数据集
- `train_AllTask_twitter2015_none_multi.sh` - 无持续学习
- `train_AllTask_twitter2015_ewc_multi.sh` - EWC策略
- `train_AllTask_twitter2015_replay_multi.sh` - Experience Replay

### 单任务训练
- `train_SingleTask_mabsa_200_none_multi.sh` - MABSA单任务(简化数据集)
- `train_SingleTask_masc_200_none_multi.sh` - MASC单任务(简化数据集)
- `train_SingleTask_mate_200_none_multi.sh` - MATE单任务(简化数据集)
- `train_SingleTask_mner_200_none_multi.sh` - MNER单任务(简化数据集)

### 标签嵌入版本
- `train_AllTask_200_none_multi.sh` - 无策略 + 标签嵌入
- `train_AllTask_200_ewc_multi.sh` - EWC + 标签嵌入

## 使用方法

### 生成单个脚本
```bash
python scripts/config_templates.py --env server --task_type AllTask --dataset twitter2015 --strategy ewc --mode multi
```

### 生成标签嵌入脚本
```bash
python scripts/config_templates.py --env server --task_type AllTask --dataset twitter2015 --strategy ewc --mode multi --use_label_embedding
```

### 批量生成所有脚本
```bash
python scripts/generate_all_scripts.py
```

## 脚本特点

### 服务器版本特点
- 模型统一命名为 `1.pt`
- 所有文件存储在 `/root/autodl-tmp/` 目录
- 适合服务器环境，节省存储空间

### 本地版本特点
- 模型详细命名：`{task}_{dataset}_{strategy}.pt`
- 文件存储在当前目录
- 适合本地开发和调试

### 标签嵌入版本特点
- 支持标签嵌入和相似度正则化
- 增强标签间语义共享
- 提高模型准确率
