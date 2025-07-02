# 多任务持续学习与标签嵌入平台

## 项目简介

本项目是一个面向多任务、持续学习（Continual Learning, CL）和标签嵌入（Label Embedding）的深度学习平台，支持多种主流 CL 策略（EWC、Replay、LwF、SI、MAS、GEM、PNN、TAM-CL、MoE、CLAP4CLIP 等），并可灵活集成标签嵌入、标签语义分组、相似度正则化等先进功能，适用于多模态（文本+图像）和多任务（MABSA、MASC、MATE、MNER）场景。

---

## 项目目录结构

```
MCM/
├── continual/                # 持续学习与标签嵌入核心代码
│   ├── label_embedding.py              # 标签嵌入核心类
│   ├── label_embedding_manager.py      # 标签嵌入管理器
│   ├── ...                            # 各类持续学习方法实现
├── datasets/                 # 数据集加载与处理
├── models/                   # 模型结构与任务头
├── modules/                  # 训练、评估、损失等通用模块
├── scripts/                  # 各类训练/批量/生成/清理脚本
│   ├── config_templates.py   # 脚本模板生成
│   ├── generate_all_scripts.py
│   ├── train_refactored.py   # 主训练入口
│   ├── ...
├── data/                     # 数据集目录
├── requirements.txt          # 依赖包
├── README.md                 # 项目说明
└── ...
```

---

## 核心功能

- **多任务持续学习**：支持主流 CL 方法，防止灾难性遗忘。
- **标签嵌入与语义对齐**：全局标签嵌入、标签分组、相似度正则化，提升跨任务知识迁移。
- **多模态融合**：支持文本、图像特征融合，适配多模态任务。
- **自动化脚本生成**：一键生成本地/服务器训练脚本，批量实验高效便捷。
- **丰富的评估指标**：平均准确率、遗忘度量、前/后向迁移等持续学习指标。
- **模块化集成**：与 EWC、Replay 等持续学习策略无缝兼容。

---

## 标签嵌入（Label Embedding）功能说明

本平台实现了**全局标签嵌入**，支持跨任务标签共享和语义对齐，特别适用于多任务持续学习。

### 主要特性

- **全局标签嵌入**：为所有任务标签构建统一嵌入空间，自动处理标签映射。
- **标签分组设计**：根据任务标签语义分组，提升标签间语义共享。
- **相似度正则化**：同组标签嵌入自动靠近，支持余弦相似度约束。
- **模块化集成**：与 EWC、Replay 等持续学习策略无缝兼容。

### 典型标签分组示例

| 分组     | 包含标签                                                | 说明          |
| -------- | ------------------------------------------------------- | ------------- |
| O        | MABSA-O, MATE-O, MNER-O                                 | 非目标标签    |
| NEG      | MABSA-B-NEG, MABSA-I-NEG, MASC-NEG                      | 负向情感/否定 |
| NEU      | MABSA-B-NEU, MABSA-I-NEU, MASC-NEU                      | 中性情感      |
| POS      | MABSA-B-POS, MABSA-I-POS, MASC-POS                      | 正向情感/肯定 |
| B_ENTITY | MATE-B, MNER-B-PER, MNER-B-ORG, MNER-B-LOC, MNER-B-MISC | 实体开始标签  |
| I_ENTITY | MATE-I, MNER-I-PER, MNER-I-ORG, MNER-I-LOC, MNER-I-MISC | 实体内部标签  |

### 使用方法（代码示例）

```python
from continual.label_embedding_manager import LabelEmbeddingManager
manager = LabelEmbeddingManager(emb_dim=128, use_similarity_regularization=True, similarity_weight=0.1)
label_embedding = manager.create_or_load_embedding(
    embedding_path="./checkpoints/label_embedding.pt", device="cuda")
```

### 训练命令行示例

```bash
python scripts/train_refactored.py \
    --task_name mabsa \
    --dataset_name twitter2015 \
    --data_dir ./data \
    --session_name session_1 \
    --use_label_embedding \
    --label_emb_dim 128 \
    --use_similarity_reg \
    --similarity_weight 0.1 \
    --label_embedding_path ./checkpoints/label_embedding.pt \
    --output_model_path ./checkpoints/model.pt \
    --train_info_json ./checkpoints/train_info.json
```

### 优势与注意事项

- **知识迁移**：相似标签在不同任务间共享语义信息，提升泛化能力。
- **标签一致性**：相似度正则化保持标签语义一致，减少冲突。
- **可扩展性**：新任务仅需扩展标签映射表。
- **兼容性**：与主流持续学习方法完全兼容。
- **注意**：训练/推理需保持标签映射一致，合理调整相似度权重，标签嵌入会增加一定内存开销。

### 扩展用法

- **预训练标签嵌入**：可用 BERT 等预训练词向量初始化标签嵌入。
- **动态标签分组**：可根据标签共现动态调整分组。
- **标签注意力机制**：可在模型中集成标签注意力头。

---

## 安装与依赖

建议使用 Python 3.8+，推荐 conda 虚拟环境。

```bash
pip install -r requirements.txt
```

**主要依赖包：**

```
matplotlib==3.8.0
numpy==2.2.4
openpyxl==3.1.5
Pillow==10.0.1
Pillow==11.1.0
pytorch_crf==0.7.2
scikit_learn==1.6.1
torch==2.1.2+cu121
torchvision==0.16.2+cu121
transformers==4.30.2
```

---

## 数据集说明

- **MABSA/MASC/MATE**：共享同一数据集（`data/MASC/twitter2015/`、`data/MASC/twitter2017/`）
- **MNER**：独立数据集（`data/MNER/twitter2015/`、`data/MNER/twitter2017/`）
- **简化数据集**：`200`样本，适合本地快速测试

---

## 快速上手

### 1. 一键生成所有训练脚本

```bash
python scripts/generate_all_scripts.py
```

### 2. 常用训练脚本

- 服务器多任务训练（EWC+标签嵌入）：
  ```bash
  ./scripts/strain_AllTask_twitter2015_ewc_multi.sh
  ```
- 本地简化数据集测试：
  ```bash
  ./scripts/train_AllTask_200_none_multi.sh
  ./scripts/train_AllTask_200_ewc_multi.sh
  ```

### 3. 自定义脚本生成

```bash
python scripts/config_templates.py \
    --env server \
    --task_type AllTask \
    --dataset twitter2015 \
    --strategy ewc \
    --mode multi \
    --use_label_embedding
```

### 4. 直接命令行训练（以标签嵌入为例）

```bash
python scripts/train_refactored.py \
    --task_name mabsa \
    --dataset_name twitter2015 \
    --data_dir ./data \
    --session_name session_1 \
    --use_label_embedding \
    --label_emb_dim 128 \
    --use_similarity_reg \
    --similarity_weight 0.1 \
    --label_embedding_path ./checkpoints/label_embedding.pt \
    --output_model_path ./checkpoints/model.pt \
    --train_info_json ./checkpoints/train_info.json
```

---

## 训练与评估流程

1. **初始化标签嵌入管理器**（可选）
2. **模型训练**：支持多种 CL 策略与标签嵌入联合优化
3. **评估与指标**：自动记录各任务准确率、遗忘度、迁移性等
4. **保存与加载**：模型、标签嵌入、EWC 参数等均可持久化

---

## 参数说明（部分）

| 参数                   | 类型  | 默认值 | 说明                           |
| ---------------------- | ----- | ------ | ------------------------------ |
| --task_name            | str   | 必填   | 任务名（mabsa/masc/mate/mner） |
| --session_name         | str   | 必填   | 训练会话名                     |
| --train_info_json      | str   | 必填   | 训练信息记录路径               |
| --output_model_path    | str   | 必填   | 模型保存路径                   |
| --data_dir             | str   | 必填   | 数据目录                       |
| --dataset_name         | str   | 必填   | 数据集名                       |
| --use_label_embedding  | flag  | False  | 启用标签嵌入                   |
| --label_emb_dim        | int   | 128    | 标签嵌入维度                   |
| --use_similarity_reg   | flag  | True   | 启用相似度正则化               |
| --similarity_weight    | float | 0.1    | 相似度正则化权重               |
| --label_embedding_path | str   | None   | 标签嵌入保存/加载路径          |
| --ewc                  | int   | 0      | 是否启用 EWC                   |
| --replay               | int   | 0      | 是否启用 Replay                |
| --lwf                  | int   | 0      | 是否启用 LwF                   |
| --si                   | int   | 0      | 是否启用 SI                    |
| --mas                  | int   | 0      | 是否启用 MAS                   |
| --gem                  | int   | 0      | 是否启用 GEM                   |
| --moe_adapters         | int   | 0      | 是否启用 MoE                   |
| --clap4clip            | int   | 0      | 是否启用 CLAP4CLIP             |
| --epochs               | int   | 20     | 训练轮数                       |
| --batch_size           | int   | 8      | 批大小                         |
| --lr                   | float | 5e-5   | 学习率                         |

> 更多参数详见 `modules/parser.py`。

---

## 脚本索引与自动化

- **批量生成所有脚本**：`python scripts/generate_all_scripts.py`
- **单个脚本生成**：
  ```bash
  python scripts/config_templates.py --env server --task_type AllTask --dataset twitter2015 --strategy ewc --mode multi
  ```
- **标签嵌入脚本**：
  ```bash
  python scripts/config_templates.py --env server --task_type AllTask --dataset twitter2015 --strategy ewc --mode multi --use_label_embedding
  ```
- **常用脚本**：
  - `strain_AllTask_twitter2015_ewc_multi.sh`（服务器 EWC+标签嵌入）
  - `train_AllTask_200_none_multi.sh`（本地无 CL 简化数据集）
  - `train_SingleTask_mabsa_200_none_multi.sh`（本地单任务）

---

## FAQ & 常见问题

- **Q: 如何恢复旧脚本？**
  ```bash
  cp scripts/backup_old_scripts/旧脚本名.sh scripts/
  chmod +x scripts/旧脚本名.sh
  ```
- **Q: 如何修改脚本参数？**
  编辑生成的脚本文件，或用`config_templates.py`重新生成
- **Q: 如何添加新 CL 策略？**
  修改`scripts/config_templates.py`中的`strategies`配置
- **Q: 脚本运行失败怎么办？**
  1. 检查 CUDA 环境：`nvidia-smi`
  2. 检查 Python 环境：`python --version`
  3. 检查依赖：`pip list | grep torch`
  4. 查看日志：`tail -f ./log/*.log`

---

## 贡献与联系

- 欢迎通过 GitHub Issue 提交建议与 Bug
- 邮箱：2120240827@main.nankai.edu.cn

---

_最后更新：2025 年 7 月_
