# t-SNE可视化设计文档

## 概述

本文档说明在多模态持续学习项目中如何使用t-SNE可视化不同类型的评估指标。

## 两种可视化类型

### 1. Token-level Micro F1 可视化

**目的**: 可视化token级别的特征分布和标签聚类

**数据点**: 每个点代表一个**token**

**颜色编码**: 实体标签（如 O, B-PER, I-PER, B-LOC等）

**应用场景**:
- MATE任务: O, B-ASPECT, I-ASPECT
- MNER任务: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC
- MABSA任务: O, B-POS, I-POS, B-NEU, I-NEU, B-NEG, I-NEG

**特征来源**:
- 从模型的中间层（如BiLSTM输出或文本编码器输出）提取每个token的特征向量
- 特征维度: 通常是 768 (BERT) 或 bilstm_hidden_size * 2

**可视化效果**:
```
    ┌─────────────────────────────────────┐
    │  Token-level Feature Distribution  │
    │                                     │
    │     ●●●  O (Non-entity)             │
    │   ●●●●●●                            │
    │  ●●●●●●●                            │
    │                                     │
    │        ▲▲▲  B-PER                   │
    │       ▲▲▲▲▲                         │
    │                                     │
    │           ■■■  I-PER                │
    │          ■■■■                       │
    │                                     │
    │  ♦♦♦  B-LOC         ◆◆◆  I-LOC     │
    │ ♦♦♦♦                ◆◆◆◆           │
    └─────────────────────────────────────┘
```

**期望观察**:
- 同类型标签的token应该聚在一起
- B- 和 I- 标签应该相近但可区分
- O标签（非实体）应该与实体标签分离
- 预测错误的token会在类别边界附近

**实现要点**:
```python
# 伪代码
features = []  # 每个token的特征向量
labels = []    # 每个token的真实标签
predictions = [] # 每个token的预测标签

for batch in dataloader:
    # 获取模型中间层输出
    token_features = model.get_token_features(batch)  # [batch_size, seq_len, hidden_dim]
    
    for seq_features, seq_labels, seq_preds in zip(token_features, batch.labels, predictions):
        for token_feat, token_label, token_pred in zip(seq_features, seq_labels, seq_preds):
            if token_label != 0:  # 排除padding和O标签（可选）
                features.append(token_feat.cpu().numpy())
                labels.append(token_label)
                # 可以根据预测正确性设置不同的marker
                predictions.append(token_pred)

# t-SNE降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedded = tsne.fit_transform(np.array(features))

# 绘图
plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='tab10')
```

---

### 2. Span-level F1 (Chunk F1) 可视化

**目的**: 可视化实体span的识别分布和准确性

**数据点**: 每个点代表一个**实体span**（完整的实体，如 "New York" 或 "Apple Inc."）

**颜色编码**: 
- **主要维度**: 实体类型（PER, LOC, ORG, MISC等，或ASPECT, POS, NEU, NEG等）
- **辅助维度**: 预测准确性
  - 正确预测: 实心圆 ●
  - 错误预测: 空心圆 ○
  - 部分正确（边界错误）: 半圆 ◐

**应用场景**:
- MATE: ASPECT实体span
- MNER: PER, LOC, ORG, MISC实体span
- MABSA: POS, NEU, NEG情感实体span

**特征来源**:
- 对实体span内所有token的特征求平均或池化
- 或使用span的首尾token特征拼接
- 或使用span-level representation（如果模型有的话）

**可视化效果**:
```
    ┌─────────────────────────────────────┐
    │  Entity Span Distribution           │
    │                                     │
    │     ●●●  PER (Correct)              │
    │   ●●●●●                             │
    │  ●○○○  PER (Incorrect)              │
    │                                     │
    │        ■■■  LOC (Correct)           │
    │       ■■○                           │
    │                                     │
    │  ▲▲▲  ORG (Correct)                 │
    │ ▲▲○                                 │
    │                                     │
    │         ♦♦♦  MISC                   │
    │        ♦♦○                          │
    └─────────────────────────────────────┘
```

**期望观察**:
- 同类型实体span应该聚在一起
- 正确预测的span（实心）应该更紧密聚类
- 错误预测的span（空心）可能在类别边界或远离聚类中心
- 不同实体类型应该有清晰的分离

**实现要点**:
```python
# 伪代码
features = []  # 每个span的特征向量
span_types = []  # 每个span的类型
span_correctness = []  # 预测是否正确

for batch in dataloader:
    token_features = model.get_token_features(batch)  # [batch_size, seq_len, hidden_dim]
    
    # 提取预测和真实的span
    pred_spans = extract_spans(batch.predictions)  # [(start, end, type), ...]
    gold_spans = extract_spans(batch.labels)       # [(start, end, type), ...]
    
    for seq_idx, (seq_features, seq_pred_spans, seq_gold_spans) in enumerate(
        zip(token_features, pred_spans, gold_spans)):
        
        # 只处理真实存在的span（用于分析recall）
        for start, end, span_type in seq_gold_spans:
            # 计算span特征（平均池化）
            span_feat = seq_features[start:end+1].mean(dim=0)
            features.append(span_feat.cpu().numpy())
            span_types.append(span_type)
            
            # 检查是否被正确预测
            is_correct = (start, end, span_type) in seq_pred_spans
            span_correctness.append(is_correct)

# t-SNE降维
tsne = TSNE(n_components=2, perplexity=20, random_state=42)
embedded = tsne.fit_transform(np.array(features))

# 绘图（根据类型和正确性）
for span_type in unique_types:
    mask = (span_types == span_type)
    correct_mask = mask & span_correctness
    incorrect_mask = mask & (~span_correctness)
    
    # 正确的用实心
    plt.scatter(embedded[correct_mask, 0], embedded[correct_mask, 1], 
                marker='o', label=f'{span_type} (correct)')
    # 错误的用空心
    plt.scatter(embedded[incorrect_mask, 0], embedded[incorrect_mask, 1], 
                marker='o', facecolors='none', edgecolors=color, 
                label=f'{span_type} (incorrect)')
```

---

## 关键区别对比

| 维度 | Token-level (Micro F1) | Span-level (Chunk F1) |
|------|------------------------|----------------------|
| **数据点单位** | 单个token | 完整实体span |
| **数据点数量** | 大（所有token） | 小（只有实体） |
| **颜色编码** | Token标签（BIO格式） | 实体类型 |
| **辅助编码** | 可选：预测正确性 | 推荐：预测正确性 |
| **特征来源** | Token-level特征 | Span-level聚合特征 |
| **关注点** | Token分类边界 | 实体识别准确性 |
| **典型用途** | 调试模型的token级别表示 | 评估实体识别效果 |

## 实现优先级

### 第一阶段：Span-level可视化（推荐优先）
**原因**:
1. 直接对应Chunk F1指标（主要评估指标）
2. 数据点少，计算快
3. 更直观地展示实体识别效果
4. 对用户更有实际意义

**使用场景**:
- 分析哪些类型的实体容易混淆
- 找出持续学习中遗忘的实体类型
- 比较不同任务/模态下的实体表示

### 第二阶段：Token-level可视化（调试用）
**原因**:
1. 数据点多，计算慢
2. 更细粒度，适合深入调试
3. 可以发现BIO标注的问题

**使用场景**:
- 调试BiLSTM-CRF的token表示
- 分析B-和I-标签的区分能力
- 研究模型的中间层表示

## 代码集成建议

### 目录结构
```
visualize/
├── __init__.py
├── training_curves.py          # 现有：训练曲线
├── tsne_visualization.py       # 新增：t-SNE可视化
└── TSNE_VISUALIZATION_DESIGN.md # 本文档
```

### 使用接口
```python
from visualize.tsne_visualization import (
    visualize_token_distribution,
    visualize_span_distribution
)

# 在evaluate.py中调用
if args.visualize_tsne:
    # Token-level
    visualize_token_distribution(
        model=model,
        dataloader=val_loader,
        output_path="checkpoints/tsne_token.png",
        task_name=args.task_name
    )
    
    # Span-level
    visualize_span_distribution(
        model=model,
        dataloader=val_loader,
        output_path="checkpoints/tsne_span.png",
        task_name=args.task_name
    )
```

### 参数配置
在 `modules/parser.py` 中添加：
```python
parser.add_argument('--visualize_tsne', type=int, default=0,
                   help='是否生成t-SNE可视化（0/1）')
parser.add_argument('--tsne_type', type=str, default='span',
                   choices=['token', 'span', 'both'],
                   help='t-SNE可视化类型')
parser.add_argument('--tsne_sample_size', type=int, default=1000,
                   help='t-SNE可视化的样本数量（避免过多数据点）')
```

## 注意事项

### 性能考虑
1. **采样**: 对于大数据集，只采样部分数据进行可视化（如1000-5000个点）
2. **缓存特征**: 提取特征后缓存，避免重复计算
3. **GPU内存**: 特征提取时注意batch size

### 可视化质量
1. **Perplexity**: 根据数据点数量调整（通常5-50）
2. **颜色方案**: 使用色盲友好的配色
3. **Legend**: 清晰标注每种颜色/形状的含义
4. **标题**: 包含任务名、数据集、epoch等信息

### 持续学习扩展
在持续学习场景中，可以：
1. **对比不同任务**: 在同一张图中用不同颜色显示不同任务的数据
2. **追踪遗忘**: 对比同一任务在不同训练阶段的分布变化
3. **迁移分析**: 观察新任务对旧任务表示空间的影响

## 参考文献

1. van der Maaten, L., & Hinton, G. (2008). "Visualizing Data using t-SNE". Journal of Machine Learning Research.
2. McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction". arXiv:1802.03426.

## TODO

- [ ] 实现 `visualize/tsne_visualization.py`
- [ ] 添加UMAP作为t-SNE的替代（更快）
- [ ] 集成到训练脚本中
- [ ] 添加交互式可视化（使用plotly）
- [ ] 支持持续学习的对比可视化

