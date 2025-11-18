# 多模态模型推理指南

## 📋 概述

`inference_complete.py` 是基于训练流程（`train_with_zero_shot.py`）设计的完整推理接口，支持所有训练的任务。

### 支持的任务

| 任务 | 类型 | 输入 | 输出 |
|------|------|------|------|
| **MATE** | 序列标注 | 文本 + 图像 | 方面术语位置和文本 |
| **MNER** | 序列标注 | 文本 + 图像 | 命名实体（PER/ORG/LOC/MISC） |
| **MABSA** | 序列标注 | 文本 + 图像 | 方面术语 + 情感（POS/NEU/NEG） |
| **MASC** | 句子分类 | 文本 + 方面词 + 图像 | 情感（-1/0/1） |

---

## 🗂️ 训练过程中保存的文件

训练脚本会保存以下文件（以`twitter2015_none_t2m_seq1`为例）：

```
checkpoints/
├── twitter2015_none_t2m_seq1.pt                      # 完整模型 ⭐ 必需
├── twitter2015_none_t2m_seq1_task_heads.pt           # 任务头（可选）
├── train_info_twitter2015_none_t2m_seq1.json         # 训练信息 ⭐ 必需
└── label_embedding_twitter2015_none_t2m_seq1.pt      # 标签嵌入（如果使用）
```

### 推理需要的文件

✅ **必需：**
1. **模型文件**：`{base_name}.pt`
2. **训练信息**：`train_info_{base_name}.json`

❌ **可选：**
- `{base_name}_task_heads.pt` - 如果模型文件中已包含任务头，则不需要
- `label_embedding_{base_name}.pt` - 推理时不需要

---

## 🚀 快速开始

### 1. MASC（句子级情感分类）

```bash
python scripts/inference_complete.py \
    --model_path checkpoints/twitter2015_none_t2m_seq1.pt \
    --train_info_path checkpoints/train_info_twitter2015_none_t2m_seq1.json \
    --task masc \
    --text "The $T$ is great but service sucks" \
    --aspect "food" \
    --image data/twitter2015/images/12345.jpg
```

**输出示例：**
```
================================================================================
预测结果（MASC - 句子级分类）
================================================================================
文本: The $T$ is great but service sucks
方面词: food
情感: positive (1)
置信度: 0.8923

概率分布:
  negative: 0.0512
  neutral: 0.0565
  positive: 0.8923
================================================================================
```

### 2. MATE（方面术语提取）

```bash
python scripts/inference_complete.py \
    --model_path checkpoints/twitter2015_none_t2m_seq1.pt \
    --train_info_path checkpoints/train_info_twitter2015_none_t2m_seq1.json \
    --task mate \
    --text "The food is great but service sucks" \
    --image data/twitter2015/images/12345.jpg
```

**输出示例：**
```
================================================================================
预测结果（MATE - 序列标注）
================================================================================
文本: The food is great but service sucks

识别的实体:
  [4:8] ENTITY: food
  [24:31] ENTITY: service

Token级别预测:
  The -> O
  food -> B
  is -> O
  great -> O
  but -> O
  service -> B
  sucks -> I
================================================================================
```

### 3. MNER（命名实体识别）

```bash
python scripts/inference_complete.py \
    --model_path checkpoints/twitter2015_none_t2m_seq1.pt \
    --train_info_path checkpoints/train_info_twitter2015_none_t2m_seq1.json \
    --task mner \
    --text "Barack Obama visited New York yesterday" \
    --image data/twitter2015/images/12345.jpg
```

**输出示例：**
```
================================================================================
预测结果（MNER - 序列标注）
================================================================================
文本: Barack Obama visited New York yesterday

识别的实体:
  [0:12] PER: Barack Obama
  [21:29] LOC: New York
================================================================================
```

### 4. MABSA（方面情感分析）

```bash
python scripts/inference_complete.py \
    --model_path checkpoints/twitter2015_none_t2m_seq1.pt \
    --train_info_path checkpoints/train_info_twitter2015_none_t2m_seq1.json \
    --task mabsa \
    --text "The food is great but service sucks" \
    --image data/twitter2015/images/12345.jpg
```

**输出示例：**
```
================================================================================
预测结果（MABSA - 序列标注）
================================================================================
文本: The food is great but service sucks

识别的实体:
  [4:8] POS: food
  [24:31] NEG: service
================================================================================
```

---

## 📖 Python API使用

### 示例1：MASC情感分类

```python
from scripts.inference_complete import MultimodalInference

# 创建推理器
predictor = MultimodalInference(
    model_path="checkpoints/twitter2015_none_t2m_seq1.pt",
    train_info_path="checkpoints/train_info_twitter2015_none_t2m_seq1.json",
    task_name="masc",
    session_name="twitter2015_masc_multimodal"  # 可选，自动推断
)

# 预测
result = predictor.predict_sentence(
    text="The $T$ is amazing",
    aspect="food",
    image_path="data/twitter2015/images/12345.jpg"
)

print(f"情感: {result['sentiment_name']}")
print(f"置信度: {result['confidence']:.4f}")
```

### 示例2：MATE实体提取

```python
from scripts.inference_complete import MultimodalInference

# 创建推理器
predictor = MultimodalInference(
    model_path="checkpoints/twitter2015_none_t2m_seq1.pt",
    train_info_path="checkpoints/train_info_twitter2015_none_t2m_seq1.json",
    task_name="mate"
)

# 预测
result = predictor.predict_sequence(
    text="The food is great but service sucks",
    image_path="data/twitter2015/images/12345.jpg",
    return_tokens=True
)

# 打印实体
for start, end, label, text in result['entities']:
    print(f"{label}: {text} [{start}:{end}]")

# 打印token级别预测
for token, label in result['token_predictions']:
    print(f"{token} -> {label}")
```

### 示例3：批量预测

```python
from scripts.inference_complete import MultimodalInference
import json

# 创建推理器
predictor = MultimodalInference(
    model_path="checkpoints/twitter2015_none_t2m_seq1.pt",
    train_info_path="checkpoints/train_info_twitter2015_none_t2m_seq1.json",
    task_name="mate"
)

# 批量数据
samples = [
    {"text": "The food is great", "image": "data/images/1.jpg"},
    {"text": "Nice restaurant", "image": "data/images/2.jpg"},
    # ... 更多样本
]

# 批量预测
results = []
for sample in samples:
    result = predictor.predict_sequence(
        text=sample['text'],
        image_path=sample['image']
    )
    results.append(result)

# 保存结果
with open("predictions.json", 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
```

---

## 🔧 高级用法

### 1. 指定会话名称

如果训练信息中有多个会话，可以指定：

```python
predictor = MultimodalInference(
    model_path="checkpoints/model.pt",
    train_info_path="checkpoints/train_info.json",
    task_name="mate",
    session_name="twitter2015_mate_text_only"  # 使用特定会话
)
```

### 2. 使用CPU推理

```python
predictor = MultimodalInference(
    model_path="checkpoints/model.pt",
    train_info_path="checkpoints/train_info.json",
    task_name="mate",
    device="cpu"  # 强制使用CPU
)
```

### 3. 处理无图像情况

如果模型是text_only模式训练的：

```python
# 模型会自动处理，提供任意图像路径或创建零张量
result = predictor.predict_sequence(
    text="The food is great",
    image_path="data/dummy.jpg"  # 即使不存在也会自动处理
)
```

---

## 📝 返回格式详解

### MASC返回格式

```python
{
    'text': str,                    # 输入文本
    'aspect': str,                  # 方面词
    'sentiment': int,               # -1(负), 0(中), 1(正)
    'sentiment_name': str,          # 'negative', 'neutral', 'positive'
    'probabilities': {
        'negative': float,
        'neutral': float,
        'positive': float
    },
    'confidence': float             # 最高概率
}
```

### 序列标注任务返回格式（MATE/MNER/MABSA）

```python
{
    'text': str,                    # 输入文本
    'entities': [                   # 识别的实体列表
        (start_pos, end_pos, label, entity_text),
        ...
    ],
    'token_predictions': [          # Token级别预测（如果return_tokens=True）
        (token, label),
        ...
    ]
}
```

---

## ⚠️ 注意事项

### 1. 模型和训练信息必须匹配

确保模型文件和训练信息文件来自同一次训练：

```bash
# ✅ 正确：同一个base_name
--model_path checkpoints/twitter2015_none_t2m_seq1.pt
--train_info_path checkpoints/train_info_twitter2015_none_t2m_seq1.json

# ❌ 错误：不同的base_name
--model_path checkpoints/twitter2015_none_t2m_seq1.pt
--train_info_path checkpoints/train_info_twitter2017_moe_t2m_seq1.json
```

### 2. 任务名称必须存在于训练信息中

```python
# 检查可用的任务
with open("checkpoints/train_info.json") as f:
    info = json.load(f)
    print("Available sessions:")
    for session in info['sessions']:
        print(f"  - {session['task_name']} ({session['session_name']})")
```

### 3. CRF模型的特殊处理

使用CRF训练的模型会自动检测并使用Viterbi解码：

```python
# 推理器会自动检测CRF
predictor = MultimodalInference(...)  # 自动处理CRF

# 输出会使用CRF解码而不是简单的argmax
result = predictor.predict_sequence(...)
```

### 4. 图像路径

- 图像文件必须存在且可读
- 如果图像加载失败，会使用零张量（可能影响性能）
- 对于text_only模式，图像输入会被忽略

---

## 🐛 故障排除

### 问题1：找不到session

```
ValueError: Could not find session for task 'mate' in train_info
```

**解决方法：**
```python
# 手动指定session_name
predictor = MultimodalInference(
    ...,
    session_name="twitter2015_mate_multimodal"  # 明确指定
)

# 或查看train_info.json中的可用session
```

### 问题2：维度不匹配

```
RuntimeError: size mismatch
```

**可能原因：**
- 模型和任务不匹配
- num_labels设置错误

**解决方法：**
检查train_info.json中的num_labels是否正确

### 问题3：CRF解码错误

```
ValueError: mask of the first timestep must all be on
```

**解决方法：**
这个错误已在推理器中处理，如果仍然出现，请报告issue

---

## 📊 性能优化

### 1. 批量推理

虽然当前接口是单样本的，但可以循环调用：

```python
import torch

predictor = MultimodalInference(...)

results = []
for sample in samples:
    with torch.no_grad():  # 确保不计算梯度
        result = predictor.predict_sequence(sample['text'], sample['image'])
        results.append(result)
```

### 2. 使用GPU

```python
# 自动选择GPU（如果可用）
predictor = MultimodalInference(..., device=None)  # 自动选择

# 或显式指定
predictor = MultimodalInference(..., device='cuda:0')
```

### 3. 半精度推理

```python
# 创建推理器后转换为half precision
predictor = MultimodalInference(...)
predictor.model = predictor.model.half()

# 注意：输入也需要half
# 这会加快推理但可能略微降低精度
```

---

## 🎯 总结

- ✅ 支持所有训练的任务（MATE/MNER/MABSA/MASC）
- ✅ 自动从训练信息推断配置
- ✅ 支持CRF模型的正确解码
- ✅ 灵活的API（命令行和Python）
- ✅ 详细的错误处理和日志

**推荐工作流：**
1. 训练模型 → 生成 `.pt` 和 `train_info.json`
2. 使用 `inference_complete.py` 进行推理
3. 根据需要调用Python API进行批量处理

---

**需要帮助？** 查看代码中的docstring或提issue！

