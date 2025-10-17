"""
MASC任务诊断工具

用于诊断MASC任务训练失败的原因，包括：
1. 数据分布分析
2. 类别权重合理性检查
3. 训练过程中的预测分布
4. 损失函数和梯度分析
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
import sys
import argparse
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.get_dataset import get_dataset
from continual.label_config import get_label_manager


def analyze_data_distribution(task_name="masc", dataset_name="twitter2015", split="train"):
    """分析数据分布"""
    print("\n" + "="*80)
    print(f"📊 分析 {task_name.upper()} - {dataset_name} - {split} 数据集分布")
    print("="*80)
    
    # 创建参数对象
    # 根据任务和数据集确定文件路径
    # 尝试多个可能的路径
    possible_paths = [
        f"data/{dataset_name}",  # 原始路径
        f"data/MASC/{dataset_name}",  # MASC子目录
    ]
    
    data_base_path = None
    for path in possible_paths:
        if os.path.exists(f"{path}/{split}.txt") or os.path.exists(f"{path}/{split}__.txt"):
            data_base_path = path
            break
    
    if data_base_path is None:
        data_base_path = possible_paths[0]  # 默认使用第一个
    
    print(f"使用数据路径: {data_base_path}")
    
    # 检查文件是否存在
    text_files = {}
    for split_name in ['train', 'dev', 'test']:
        for suffix in ['', '__']:
            file_path = f"{data_base_path}/{split_name}{suffix}.txt"
            if os.path.exists(file_path):
                text_files[f"{split_name}_text_file"] = file_path
                break
        if f"{split_name}_text_file" not in text_files:
            # 默认路径
            text_files[f"{split_name}_text_file"] = f"{data_base_path}/{split_name}.txt"
    
    args = argparse.Namespace(
        task_name=task_name,
        dataset=dataset_name,
        image_dir=f"{data_base_path}/images",
        **text_files,
        text_model_name="microsoft/deberta-v3-base",
        max_seq_length=128,
        deqa=False
    )
    
    try:
        # 直接读取文本文件来统计标签，避免加载图片
        text_file = text_files.get(f"{split}_text_file")
        if not os.path.exists(text_file):
            print(f"❌ 文本文件不存在: {text_file}")
            return None, None
        
        print(f"读取文本文件: {text_file}")
        
        # 统计标签分布（直接从文件读取，不加载数据集）
        # MASC 数据格式为每 4 行一个样本：
        # 1) 原文，带 $T$ 占位符
        # 2) aspect_term (替换 $T$ 的真实字符串)
        # 3) sentiment (可能是 -1, 0, 1)
        # 4) image_name (图像文件名)
        labels_list = []
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines()]
        
        # 检查格式
        if len(lines) % 4 != 0:
            print(f"❌ 数据格式错误：文件行数 {len(lines)} 不是4的倍数")
            return None, None
        
        # 每4行为一组
        for i in range(0, len(lines), 4):
            text_with_T = lines[i]
            aspect_term = lines[i+1]
            sentiment_str = lines[i+2]
            image_name = lines[i+3]
            
            try:
                sentiment = int(sentiment_str)  # -1, 0, 1
                # 映射到标签ID: -1->0(NEG), 0->1(NEU), 1->2(POS)
                label_id = sentiment + 1
                labels_list.append(label_id)
            except ValueError:
                print(f"⚠️  警告：无法解析情感值 '{sentiment_str}' (行 {i+3})")
                continue
        
        # 计算分布
        label_counter = Counter(labels_list)
        total = len(labels_list)
        
        if total == 0:
            print("❌ 没有读取到任何样本！")
            return None, None
        
        # 获取标签名称
        label_manager = get_label_manager()
        task_config = label_manager.get_task_config(task_name)
        label_names = task_config.label_names
        
        print(f"\n总样本数: {total}")
        print("\n标签分布:")
        print("-" * 60)
        print(f"{'标签ID':<10} {'标签名':<15} {'样本数':<10} {'占比':<10} {'频率倒数':<10}")
        print("-" * 60)
        
        for label_id in sorted(label_counter.keys()):
            count = label_counter[label_id]
            ratio = count / total
            inv_freq = 1.0 / ratio if ratio > 0 else 0
            label_name = label_names[label_id] if label_id < len(label_names) else f"Unknown-{label_id}"
            print(f"{label_id:<10} {label_name:<15} {count:<10} {ratio*100:>6.2f}%   {inv_freq:>8.2f}")
        
        print("-" * 60)
        
        # 计算不平衡程度
        counts = np.array([label_counter[i] for i in range(len(label_names))])
        imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
        print(f"\n⚠️  类别不平衡比: {imbalance_ratio:.2f}x (最多类/最少类)")
        
        if imbalance_ratio > 10:
            print("   ❌ 严重不平衡！建议使用类别权重")
        elif imbalance_ratio > 3:
            print("   ⚠️  中度不平衡，建议使用类别权重")
        else:
            print("   ✅ 相对平衡")
        
        return label_counter, label_names
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def analyze_class_weights(task_name="masc"):
    """分析当前配置的类别权重"""
    print("\n" + "="*80)
    print(f"⚖️  分析 {task_name.upper()} 类别权重配置")
    print("="*80)
    
    label_manager = get_label_manager()
    task_config = label_manager.get_task_config(task_name)
    label_names = task_config.label_names
    
    # 获取类别权重
    device = torch.device("cpu")
    class_weights = label_manager.get_class_weights(task_name, device)
    
    if class_weights is None:
        print("❌ 当前没有配置类别权重！")
        return None
    
    print(f"\n当前类别权重:")
    print("-" * 40)
    for i, (name, weight) in enumerate(zip(label_names, class_weights)):
        print(f"{i}. {name:<15}: {weight.item():.2f}")
    print("-" * 40)
    
    # 分析权重比例
    weight_ratio = class_weights.max() / class_weights.min()
    print(f"\n权重比例: {weight_ratio:.2f}x (最大/最小)")
    
    if weight_ratio > 20:
        print("⚠️  警告: 权重差距过大，可能导致训练不稳定")
    elif weight_ratio < 2:
        print("⚠️  警告: 权重差距太小，可能无法有效缓解类别不平衡")
    else:
        print("✅ 权重比例合理")
    
    return class_weights


def analyze_loss_function(task_name="masc"):
    """分析损失函数的计算"""
    print("\n" + "="*80)
    print(f"📐 分析损失函数 - {task_name.upper()}")
    print("="*80)
    
    label_manager = get_label_manager()
    task_config = label_manager.get_task_config(task_name)
    
    print(f"\n任务类型: {task_config.task_type.value}")
    print(f"标签数量: {task_config.num_labels}")
    print(f"标签名称: {task_config.label_names}")
    
    # 检查分类器结构
    print("\n分类器结构:")
    print(f"  • 输入维度: hidden_dim (通常768)")
    print(f"  • 输出维度: {task_config.num_labels}")
    print(f"  • 分类器类型: {'三分类' if task_config.num_labels == 3 else f'{task_config.num_labels}分类'}")
    
    # 检查损失函数
    device = torch.device("cpu")
    class_weights = label_manager.get_class_weights(task_name, device)
    
    print("\n损失函数配置:")
    if class_weights is not None:
        print(f"  ✅ 使用类别权重: F.cross_entropy(logits, labels, weight=class_weights)")
        print(f"  • 权重值: {[f'{w:.2f}' for w in class_weights]}")
    else:
        print(f"  ❌ 未使用类别权重: F.cross_entropy(logits, labels)")
        print(f"  ⚠️  这可能导致模型偏向多数类别！")
    
    # 模拟损失计算
    print("\n模拟损失计算示例:")
    print("-" * 60)
    
    # 假设的logits和labels
    logits = torch.tensor([[2.0, 3.0, 1.5], [1.0, 4.0, 2.0], [2.5, 2.0, 3.5]])  # 3个样本，3类
    labels = torch.tensor([0, 1, 2])  # NEG, NEU, POS各一个
    
    # 不使用权重
    loss_no_weight = F.cross_entropy(logits, labels)
    print(f"不使用权重的损失: {loss_no_weight.item():.4f}")
    
    # 使用权重
    if class_weights is not None:
        loss_with_weight = F.cross_entropy(logits, labels, weight=class_weights)
        print(f"使用权重的损失: {loss_with_weight.item():.4f}")
        print(f"差异: {abs(loss_with_weight.item() - loss_no_weight.item()):.4f}")
    
    print("-" * 60)


def recommend_class_weights(label_counter, label_names, strategy="balanced"):
    """推荐类别权重
    
    Args:
        label_counter: 标签计数器
        label_names: 标签名称列表
        strategy: 权重策略 ("balanced", "sqrt", "log", "inverse")
    """
    print("\n" + "="*80)
    print(f"💡 推荐类别权重 (策略: {strategy})")
    print("="*80)
    
    total = sum(label_counter.values())
    if total == 0:
        print("❌ 没有可用的样本数据，无法推荐权重")
        return
    
    n_classes = len(label_names)
    
    recommended_weights = []
    
    for i in range(n_classes):
        count = label_counter.get(i, 1)  # 避免除零
        freq = count / total
        
        if strategy == "balanced":
            # sklearn的balanced策略: n_samples / (n_classes * n_samples_per_class)
            weight = total / (n_classes * count)
        elif strategy == "sqrt":
            # 平方根倒数
            weight = 1.0 / np.sqrt(freq)
        elif strategy == "log":
            # 对数倒数
            weight = 1.0 / np.log(freq + 1e-6)
        elif strategy == "inverse":
            # 简单倒数
            weight = 1.0 / freq
        else:
            weight = 1.0
        
        recommended_weights.append(weight)
    
    # 归一化权重（将最小权重设为1.0）
    recommended_weights = np.array(recommended_weights)
    recommended_weights = recommended_weights / recommended_weights.min()
    
    # 限制最大权重比例（避免过大）
    max_ratio = 20.0
    if recommended_weights.max() / recommended_weights.min() > max_ratio:
        recommended_weights = np.clip(recommended_weights, 1.0, max_ratio)
        print(f"⚠️  权重已裁剪到最大比例 {max_ratio}x")
    
    print("\n推荐权重:")
    print("-" * 60)
    print(f"{'标签':<15} {'样本数':<10} {'频率':<10} {'推荐权重':<10}")
    print("-" * 60)
    
    for i, name in enumerate(label_names):
        count = label_counter.get(i, 0)
        freq = count / total
        weight = recommended_weights[i]
        print(f"{name:<15} {count:<10} {freq*100:>6.2f}%   {weight:>8.2f}")
    
    print("-" * 60)
    print(f"\n在 continual/label_config.py 中使用:")
    weight_str = ", ".join([f"{w:.1f}" for w in recommended_weights])
    print(f'    "masc": [{weight_str}],  # {label_names}')
    
    return recommended_weights


def main():
    parser = argparse.ArgumentParser(description="MASC任务诊断工具")
    parser.add_argument("--task", type=str, default="masc", help="任务名称")
    parser.add_argument("--dataset", type=str, default="twitter2015", help="数据集名称")
    parser.add_argument("--split", type=str, default="train", help="数据集划分")
    parser.add_argument("--recommend", type=str, default="balanced", 
                       choices=["balanced", "sqrt", "log", "inverse"],
                       help="推荐权重策略")
    
    args = parser.parse_args()
    
    # 1. 分析数据分布
    label_counter, label_names = analyze_data_distribution(args.task, args.dataset, args.split)
    
    if label_counter is None:
        return
    
    # 2. 分析当前权重
    analyze_class_weights(args.task)
    
    # 3. 分析损失函数
    analyze_loss_function(args.task)
    
    # 4. 推荐权重
    if args.recommend:
        recommend_class_weights(label_counter, label_names, args.recommend)
    
    # 打印诊断建议
    print("\n" + "="*80)
    print("📋 MASC训练失败诊断建议")
    print("="*80)
    print("""
🔴 常见问题: MASC训练时模型全部预测NEU或POS，无法学习NEG

原因分析:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 严重的类别不平衡 (NEU占60%, POS占30%, NEG仅占10%)
2. 分类器是标准的3分类器: Linear(hidden_dim, 3) 
3. 损失函数如果不使用类别权重,会严重偏向多数类

解决方案:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 方案1: 调整类别权重 (首选)
   当前权重: [5.0, 1.0, 2.0]  # [NEG, NEU, POS]
   
   建议尝试:
   a) 更激进的权重: [10.0, 1.0, 3.0]  # 大幅提升NEG
   b) Balanced权重: 使用 --recommend balanced 查看推荐值
   c) 动态调整: 训练前几个epoch用高权重,后续逐渐降低

✅ 方案2: 降低学习率
   - 当前学习率如果是5e-5,改为1e-5或5e-6
   - 更小的学习率让模型有更多时间学习minority class
   - 配合增加训练epochs (例如从10增加到20)

✅ 方案3: 减小Batch Size
   - 从32降到16或8
   - 更小的batch让模型更频繁地看到NEG样本
   - 注意相应调整学习率 (batch减半,lr也减半)

✅ 方案4: 使用Focal Loss
   在 modules/training_loop_fixed.py 中替换损失函数:
   ```python
   # 代替 F.cross_entropy
   from torch.nn import functional as F
   
   def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
       ce_loss = F.cross_entropy(logits, labels, reduction='none')
       pt = torch.exp(-ce_loss)
       focal_loss = alpha * (1-pt)**gamma * ce_loss
       return focal_loss.mean()
   ```

✅ 方案5: 过采样NEG类
   在数据加载时对NEG样本重复采样2-3次

⚠️  方案6: 检查数据质量
   - 确认NEG样本的标注是否正确
   - 检查NEG样本是否真的与NEU/POS有明显区别
   - 可视化一些NEG样本看是否合理

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 监控指标:
   训练时不要只看总体Acc,要监控每个类别的Precision/Recall:
   - NEG Recall: NEG样本中被正确预测的比例 (目标>60%)
   - NEG Precision: 预测为NEG中真正是NEG的比例 (目标>50%)
   - 如果Recall=0%, 说明模型完全没学会NEG
   - 如果Recall>0%但Precision很低, 说明模型乱猜NEG

🎯 训练技巧:
   1. 前5个epoch用高class weight [10.0, 1.0, 3.0]
   2. 监控NEG recall,如果>40%,降低weight到 [7.0, 1.0, 2.5]
   3. 最后几个epoch降到 [5.0, 1.0, 2.0]
   4. 整个过程使用低学习率 (1e-5) 和小batch (16)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()

