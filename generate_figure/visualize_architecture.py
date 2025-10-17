#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Label Embedding Guided Continual Learning for Multimodal Information Extraction
架构可视化脚本（科研论文级别）

生成高质量的架构图，包括：
1. 整体框架图
2. Label Embedding机制详细图
3. 持续学习流程图
4. 多模态融合详细图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import sys
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300  # 高分辨率
plt.rcParams['font.size'] = 10

# 颜色方案（专业配色）
COLORS = {
    'input': '#E8F4F8',        # 浅蓝 - 输入
    'encoder': '#B8E6F0',      # 蓝 - 编码器
    'fusion': '#FFE6CC',       # 橙 - 融合
    'label_emb': '#FFD9D9',    # 红 - Label Embedding
    'head': '#D9F0D9',         # 绿 - 任务头
    'output': '#F0E6FF',       # 紫 - 输出
    'cl': '#FFF4CC',           # 黄 - 持续学习
    'arrow': '#666666',        # 灰 - 箭头
    'text': '#000000'          # 黑 - 文字
}


def draw_box(ax, xy, width, height, text, color, fontsize=10, fontweight='normal'):
    """绘制带文字的方框"""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.05",
        edgecolor='black',
        facecolor=color,
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(box)
    
    # 添加文字
    ax.text(
        xy[0] + width/2, xy[1] + height/2,
        text,
        ha='center', va='center',
        fontsize=fontsize,
        fontweight=fontweight,
        zorder=3
    )
    
    return box


def draw_arrow(ax, start, end, style='->',  connectionstyle="arc3,rad=0", color='black', linewidth=2):
    """绘制箭头"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        connectionstyle=connectionstyle,
        color=color,
        linewidth=linewidth,
        zorder=1
    )
    ax.add_patch(arrow)
    return arrow


def create_overall_architecture(save_path='figures/overall_architecture.png'):
    """
    图1: 整体架构图
    展示从输入到输出的完整流程
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(7, 9.5, 'Label Embedding Guided Continual Learning\nfor Multimodal Information Extraction',
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # ========== 输入层 ==========
    y_input = 8.0
    draw_box(ax, (0.5, y_input), 2, 0.6, 'Text Input\n[CLS] w₁ w₂ ... [SEP]', COLORS['input'], fontsize=9)
    draw_box(ax, (3.5, y_input), 2, 0.6, 'Image Input\n224×224×3', COLORS['input'], fontsize=9)
    
    # ========== 编码器层 ==========
    y_encoder = 6.5
    # 文本编码器
    draw_box(ax, (0.2, y_encoder), 2.6, 0.8, 'Text Encoder\n(DeBERTa-v3)', COLORS['encoder'], fontsize=9, fontweight='bold')
    draw_arrow(ax, (1.5, y_input), (1.5, y_encoder+0.8))
    
    # 图像编码器
    draw_box(ax, (3.2, y_encoder), 2.6, 0.8, 'Image Encoder\n(ViT)', COLORS['encoder'], fontsize=9, fontweight='bold')
    draw_arrow(ax, (4.5, y_input), (4.5, y_encoder+0.8))
    
    # 特征维度标注
    ax.text(1.5, y_encoder+0.3, 'H×L×D', ha='center', fontsize=7, style='italic', color='blue')
    ax.text(4.5, y_encoder+0.3, 'H×D', ha='center', fontsize=7, style='italic', color='blue')
    
    # ========== 多模态融合层 ==========
    y_fusion = 5.0
    draw_box(ax, (1.5, y_fusion), 3.0, 0.8, 'Multimodal Fusion\n(Gated/Attention/Concat)', 
             COLORS['fusion'], fontsize=9, fontweight='bold')
    draw_arrow(ax, (1.5, y_encoder), (2.2, y_fusion+0.8))
    draw_arrow(ax, (4.5, y_encoder), (3.8, y_fusion+0.8))
    
    ax.text(3.0, y_fusion+0.3, 'F = φ(Hₜₑₓₜ, Hᵢₘₐ)', ha='center', fontsize=7, style='italic', color='blue')
    
    # ========== Label Embedding模块（核心创新） ==========
    y_label = 5.0
    label_box = draw_box(ax, (7.0, y_label-0.5), 3.5, 1.8, '', COLORS['label_emb'], fontsize=9)
    
    # Label Embedding内部结构
    ax.text(8.75, y_label+1.1, 'Label Embedding Module', ha='center', fontsize=10, fontweight='bold')
    
    # Global Label Mapping
    draw_box(ax, (7.2, y_label+0.5), 3.1, 0.4, 'Global Label Mapping\n{(task, label_id) → global_idx}', 
             '#FFFFFF', fontsize=7)
    
    # Pretrained Embeddings
    draw_box(ax, (7.2, y_label-0.1), 1.45, 0.4, 'Pretrained\nEmbeddings', '#FFFFFF', fontsize=7)
    
    # Label Groups
    draw_box(ax, (8.75, y_label-0.1), 1.45, 0.4, 'Label\nGroups', '#FFFFFF', fontsize=7)
    
    # 箭头指向
    draw_arrow(ax, (6.5, y_label+0.6), (7.2, y_label+0.6), linewidth=1.5)
    ax.text(6.0, y_label+0.7, 'Task Info', ha='center', fontsize=7)
    
    # ========== 任务特定头 ==========
    y_head = 3.5
    draw_box(ax, (1.0, y_head), 4.5, 0.8, 'Task-Specific Head with Label Attention\nLogits = TokenProj(F) · LabelProj(E_label) / √d', 
             COLORS['head'], fontsize=9, fontweight='bold')
    draw_arrow(ax, (3.0, y_fusion), (3.25, y_head+0.8))
    draw_arrow(ax, (8.75, y_label-0.5), (5.0, y_head+0.5), connectionstyle="arc3,rad=0.3", linewidth=1.5)
    
    ax.text(6.5, y_head+0.7, 'E_label', ha='center', fontsize=8, style='italic', color='red')
    
    # ========== 持续学习组件 ==========
    y_cl = 2.0
    cl_box = draw_box(ax, (6.5, y_cl), 6.5, 1.0, '', COLORS['cl'])
    ax.text(9.75, y_cl+0.7, 'Continual Learning Strategies', ha='center', fontsize=10, fontweight='bold')
    
    # 各个CL策略
    cl_methods = ['EWC', 'Replay', 'GEM', 'LwF', 'SI', 'MAS']
    x_start = 6.7
    for i, method in enumerate(cl_methods):
        x = x_start + i * 1.05
        draw_box(ax, (x, y_cl+0.1), 0.95, 0.3, method, '#FFFFFF', fontsize=7)
    
    # CL箭头
    draw_arrow(ax, (3.25, y_head), (3.25, y_cl+0.9), style='->', linewidth=1.5, color='orange')
    ax.text(3.8, y_cl+1.1, 'Regularization', ha='left', fontsize=8, color='orange')
    
    # ========== 输出层 ==========
    y_output = 0.5
    draw_box(ax, (0.5, y_output), 2.0, 0.6, 'Token-Level\nPredictions\n(MATE/MNER/MABSA)', 
             COLORS['output'], fontsize=8)
    draw_box(ax, (3.5, y_output), 2.0, 0.6, 'Sentence-Level\nPredictions\n(MASC)', 
             COLORS['output'], fontsize=8)
    
    draw_arrow(ax, (2.5, y_head), (1.5, y_output+0.6))
    draw_arrow(ax, (3.5, y_head), (4.5, y_output+0.6))
    
    # ========== 关键创新标注 ==========
    # 创新点1: Label Embedding
    innovation1 = mpatches.FancyBboxPatch(
        (11.0, y_label-0.2), 2.5, 0.8,
        boxstyle="round,pad=0.1",
        edgecolor='red',
        facecolor='white',
        linewidth=2,
        linestyle='--'
    )
    ax.add_patch(innovation1)
    ax.text(12.25, y_label+0.4, '💡 Innovation 1:', ha='center', fontsize=8, fontweight='bold', color='red')
    ax.text(12.25, y_label+0.1, 'Shared Label', ha='center', fontsize=7)
    ax.text(12.25, y_label-0.1, 'Embeddings', ha='center', fontsize=7)
    
    # 创新点2: Label Attention
    innovation2 = mpatches.FancyBboxPatch(
        (11.0, y_head+0.1), 2.5, 0.6,
        boxstyle="round,pad=0.1",
        edgecolor='red',
        facecolor='white',
        linewidth=2,
        linestyle='--'
    )
    ax.add_patch(innovation2)
    ax.text(12.25, y_head+0.5, '💡 Innovation 2:', ha='center', fontsize=8, fontweight='bold', color='red')
    ax.text(12.25, y_head+0.2, 'Label Attention', ha='center', fontsize=7)
    
    # 保存
    Path('figures').mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 整体架构图已保存到: {save_path}")
    plt.close()


def create_label_embedding_details(save_path='figures/label_embedding_details.png'):
    """
    图2: Label Embedding机制详细图
    展示如何构建和使用label embedding
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(8, 9.5, 'Label Embedding Mechanism',
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # ========== 步骤1: Label Text Definition ==========
    y_step1 = 8.0
    ax.text(1, y_step1+0.5, 'Step 1: Label Text Definition', fontsize=11, fontweight='bold')
    
    # 任务标签
    tasks_labels = [
        ('MABSA', ['O', 'B-NEG', 'I-NEG', 'B-POS', '...'], 7),
        ('MASC', ['NEG', 'NEU', 'POS'], 3),
        ('MATE', ['O', 'B', 'I'], 3),
        ('MNER', ['O', 'B-PER', 'I-PER', '...'], 9)
    ]
    
    x_task = 0.5
    for i, (task, labels, num) in enumerate(tasks_labels):
        x = x_task + i * 3.5
        # 任务框
        draw_box(ax, (x, y_step1-0.3), 3.0, 0.6, '', '#F0F0F0')
        ax.text(x+1.5, y_step1+0.15, f'{task} ({num} labels)', ha='center', fontsize=9, fontweight='bold')
        # 标签列表
        label_text = ', '.join(labels)
        ax.text(x+1.5, y_step1-0.1, label_text, ha='center', fontsize=7)
    
    # ========== 步骤2: Global Mapping ==========
    y_step2 = 6.5
    ax.text(1, y_step2+0.5, 'Step 2: Global Label Mapping', fontsize=11, fontweight='bold')
    
    # 映射表
    draw_box(ax, (0.5, y_step2-0.8), 13, 1.0, '', '#FFFACD')
    ax.text(7, y_step2+0.1, 'label2idx: {(task, label_id) → global_idx}', ha='center', fontsize=9, style='italic')
    
    # 示例映射
    examples = [
        '(MABSA, 0) → 0',
        '(MABSA, 1) → 1',
        '...',
        '(MASC, 0) → 7',
        '(MASC, 1) → 8',
        '...',
        'Total: 22 global labels'
    ]
    example_text = '  |  '.join(examples)
    ax.text(7, y_step2-0.4, example_text, ha='center', fontsize=7, family='monospace')
    
    # 箭头
    draw_arrow(ax, (7, y_step1-0.4), (7, y_step2+0.2))
    
    # ========== 步骤3: Pretrained Embeddings ==========
    y_step3 = 5.0
    ax.text(1, y_step3+0.5, 'Step 3: Generate Pretrained Embeddings', fontsize=11, fontweight='bold')
    
    # DeBERTa编码
    draw_box(ax, (1.0, y_step3-0.5), 4.0, 0.8, 'DeBERTa-v3-base Encoder', COLORS['encoder'], fontsize=9)
    ax.text(3.0, y_step3-0.1, 'Input: label text descriptions', ha='center', fontsize=7, style='italic')
    ax.text(3.0, y_step3-0.35, 'Output: E_label^(0) ∈ ℝ^(22×768)', ha='center', fontsize=8, style='italic', color='blue')
    
    # 标签组
    draw_box(ax, (6.0, y_step3-0.5), 4.0, 0.8, 'Semantic Label Groups', COLORS['label_emb'], fontsize=9)
    
    groups_text = [
        'NEG: {(MABSA,1), (MABSA,2), (MASC,0)}',
        'POS: {(MABSA,5), (MABSA,6), (MASC,2)}',
        '...'
    ]
    for i, txt in enumerate(groups_text):
        ax.text(8.0, y_step3-0.05-i*0.2, txt, ha='center', fontsize=6, family='monospace')
    
    # 箭头
    draw_arrow(ax, (7, y_step2-0.8), (3, y_step3+0.3))
    draw_arrow(ax, (7, y_step2-0.8), (8, y_step3+0.3))
    
    # ========== 步骤4: Learnable Label Embeddings ==========
    y_step4 = 3.5
    ax.text(1, y_step4+0.5, 'Step 4: Learnable Label Embeddings', fontsize=11, fontweight='bold')
    
    draw_box(ax, (1.5, y_step4-0.6), 8.0, 1.0, '', COLORS['label_emb'])
    
    # 嵌入矩阵
    ax.text(5.5, y_step4+0.2, 'E_label = nn.Embedding(22, 128)', ha='center', fontsize=9, fontweight='bold')
    ax.text(5.5, y_step4-0.05, 'Initialized with E_label^(0) (DeBERTa features)', ha='center', fontsize=7, style='italic')
    ax.text(5.5, y_step4-0.3, 'Trainable: ✓ (new task labels) | Frozen: 🔒 (old task labels)', ha='center', fontsize=7)
    
    # 相似度正则化
    ax.text(12.0, y_step4+0.2, 'Similarity', ha='center', fontsize=8, fontweight='bold')
    ax.text(12.0, y_step4-0.05, 'Regularization:', ha='center', fontsize=8, fontweight='bold')
    ax.text(12.0, y_step4-0.35, 'L_sim = Σ ||sim(e_i, e_j) - 1||²', ha='center', fontsize=7, style='italic', family='monospace')
    
    # 箭头
    draw_arrow(ax, (3, y_step3-0.5), (4, y_step4+0.4))
    draw_arrow(ax, (8, y_step3-0.5), (7, y_step4+0.4))
    
    # ========== 步骤5: Label Attention Head ==========
    y_step5 = 2.0
    ax.text(1, y_step5+0.5, 'Step 5: Label Attention Prediction', fontsize=11, fontweight='bold')
    
    draw_box(ax, (1.5, y_step5-0.6), 8.0, 0.9, '', COLORS['head'])
    
    # 公式
    ax.text(5.5, y_step5+0.15, 'Token-Level:', ha='center', fontsize=9, fontweight='bold')
    ax.text(5.5, y_step5-0.1, 'Logits = TokenProj(F) · LabelProj(E_label)^T / √d', ha='center', fontsize=8, family='monospace')
    ax.text(5.5, y_step5-0.35, 'Shape: (B, L, C) = (B, L, hidden) @ (hidden, C)^T', ha='center', fontsize=7, style='italic', color='blue')
    
    # 箭头
    # draw_arrow(ax, (3.0, y_step4), (3.0, y_step5+0.3), linewidth=1.5)  # 移除（布局问题）
    draw_arrow(ax, (5.5, y_step4-0.6), (5.5, y_step5+0.3), linewidth=1.5, color='red')
    
    # ========== 步骤6: Output ==========
    y_output = 0.5
    draw_box(ax, (3.0, y_output), 4.0, 0.6, 'Predictions: P = softmax(Logits)', 
             COLORS['output'], fontsize=9, fontweight='bold')
    draw_arrow(ax, (5.5, y_step5-0.6), (5.0, y_output+0.6))
    
    # ========== 优势标注 ==========
    advantages = [
        '✓ Cross-task knowledge sharing',
        '✓ Semantic-guided learning',
        '✓ Reduced catastrophic forgetting',
        '✓ Zero-shot transfer capability'
    ]
    
    ax.text(12.5, 1.5, 'Key Advantages:', ha='left', fontsize=9, fontweight='bold', color='darkgreen')
    for i, adv in enumerate(advantages):
        ax.text(12.5, 1.2-i*0.25, adv, ha='left', fontsize=7, color='darkgreen')
    
    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Label Embedding详细图已保存到: {save_path}")
    plt.close()


def create_continual_learning_flow(save_path='figures/continual_learning_flow.png'):
    """
    图3: 持续学习流程图
    展示多任务顺序学习过程
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 标题
    ax.text(8, 7.5, 'Continual Learning Flow',
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # 时间轴
    y_timeline = 6.5
    ax.arrow(0.5, y_timeline, 14.5, 0, head_width=0.15, head_length=0.3, fc='black', ec='black', linewidth=2)
    ax.text(8, y_timeline-0.5, 'Time →', ha='center', fontsize=10, style='italic')
    
    # 任务序列
    tasks = [
        ('Task 1\nMASC', 'text_only', COLORS['input']),
        ('Task 2\nMATE', 'text_only', COLORS['input']),
        ('Task 3\nMNER', 'text_only', COLORS['input']),
        ('Task 4\nMABSA', 'text_only', COLORS['input']),
        ('Task 5\nMASC', 'multimodal', COLORS['encoder']),
        ('Task 6\nMATE', 'multimodal', COLORS['encoder']),
        ('Task 7\nMNER', 'multimodal', COLORS['encoder']),
        ('Task 8\nMABSA', 'multimodal', COLORS['encoder'])
    ]
    
    x_start = 1.0
    task_width = 1.5
    task_spacing = 0.3
    
    for i, (task_name, modality, color) in enumerate(tasks):
        x = x_start + i * (task_width + task_spacing)
        
        # 任务框
        draw_box(ax, (x, y_timeline-0.8), task_width, 0.6, task_name, color, fontsize=8)
        
        # 模态标签
        modality_color = '#90EE90' if modality == 'text_only' else '#87CEEB'
        ax.text(x+task_width/2, y_timeline-1.1, modality, ha='center', fontsize=6, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor=modality_color, edgecolor='black', linewidth=0.5))
        
        # 任务编号
        ax.text(x+task_width/2, y_timeline+0.3, f't={i+1}', ha='center', fontsize=7, fontweight='bold')
    
    # ========== 每个任务的处理流程 ==========
    y_flow = 4.5
    
    # 选择一个任务展开详细流程（Task 5: MASC multimodal）
    task_idx = 4
    x_detail = x_start + task_idx * (task_width + task_spacing) + task_width/2
    
    # 指示线
    draw_arrow(ax, (x_detail, y_timeline-0.8), (x_detail, y_flow+1.5), style='->', linewidth=2, color='red')
    ax.text(x_detail+0.3, y_flow+1.8, 'Detailed Flow ↓', fontsize=8, color='red', fontweight='bold')
    
    # 详细流程框
    flow_box = mpatches.FancyBboxPatch(
        (x_detail-3, y_flow-3.5), 6, 4.8,
        boxstyle="round,pad=0.15",
        edgecolor='red',
        facecolor='white',
        linewidth=2,
        linestyle='-'
    )
    ax.add_patch(flow_box)
    
    # 流程步骤
    y_pos = y_flow + 1.0
    
    # 1. Load Data & Previous Model
    draw_box(ax, (x_detail-2.5, y_pos), 5, 0.4, '① Load Data + Previous Model θ_(t-1)', '#E0E0E0', fontsize=7)
    y_pos -= 0.6
    
    # 2. Freeze Old Label Embeddings
    draw_box(ax, (x_detail-2.5, y_pos), 5, 0.4, '② Freeze Old Label Embeddings E_old', COLORS['label_emb'], fontsize=7)
    y_pos -= 0.6
    
    # 3. Train with Label Attention
    draw_box(ax, (x_detail-2.5, y_pos), 5, 0.5, '③ Train: L = L_CE + L_CL + λ·L_sim', COLORS['head'], fontsize=7)
    # 损失组件
    loss_components = [
        'L_CE: Cross-Entropy',
        'L_CL: EWC/Replay/GEM...',
        'L_sim: Label Similarity'
    ]
    for j, comp in enumerate(loss_components):
        ax.text(x_detail-2.3, y_pos-0.15-j*0.15, f'• {comp}', fontsize=6)
    y_pos -= 0.95
    
    # 4. Update Label Embeddings
    draw_box(ax, (x_detail-2.5, y_pos), 5, 0.4, '④ Update New Label Embeddings E_new', COLORS['label_emb'], fontsize=7)
    y_pos -= 0.6
    
    # 5. Estimate Fisher / Update Memory
    draw_box(ax, (x_detail-2.5, y_pos), 5, 0.4, '⑤ Update CL Components (Fisher/Memory)', COLORS['cl'], fontsize=7)
    y_pos -= 0.6
    
    # 6. Zero-Shot Evaluation
    draw_box(ax, (x_detail-2.5, y_pos), 5, 0.4, '⑥ Zero-Shot Evaluation on Future Tasks', COLORS['output'], fontsize=7)
    y_pos -= 0.6
    
    # 7. Save Model
    draw_box(ax, (x_detail-2.5, y_pos), 5, 0.4, '⑦ Save Model θ_t & Label Embeddings', '#D0D0D0', fontsize=7)
    
    # ========== 知识传递图示 ==========
    # 从之前的任务到当前任务
    for i in range(max(0, task_idx-2), task_idx):
        x_prev = x_start + i * (task_width + task_spacing) + task_width/2
        draw_arrow(ax, (x_prev, y_timeline-0.8), (x_detail, y_flow+1.5), 
                  style='->', linewidth=1, color='gray', connectionstyle="arc3,rad=0.2")
    
    ax.text(x_detail-4.5, y_flow+0.5, 'Knowledge\nTransfer', ha='center', fontsize=7, color='gray', style='italic')
    
    # ========== 评估矩阵示意 ==========
    ax.text(1, 0.8, 'Performance Matrix:', fontsize=10, fontweight='bold')
    
    # 简化的准确率矩阵
    matrix_data = [
        ['', 't=1', 't=2', 't=3', '...', 't=8'],
        ['t=1', '90.2', '12.5↗', '8.3↗', '...', '5.1↗'],
        ['t=2', '88.5↘', '92.1', '15.2↗', '...', '7.8↗'],
        ['t=3', '87.9↘', '91.3↘', '85.4', '...', '9.2↗'],
        ['...', '...', '...', '...', '...', '...'],
        ['t=8', '86.2↘', '90.1↘', '84.2↘', '...', '88.9'],
    ]
    
    # 绘制矩阵
    cell_width = 1.2
    cell_height = 0.25
    x_matrix = 1.0
    y_matrix = 0.5
    
    for i, row in enumerate(matrix_data):
        for j, cell in enumerate(row):
            x_cell = x_matrix + j * cell_width
            y_cell = y_matrix - i * cell_height
            
            # 表头
            if i == 0 or j == 0:
                ax.text(x_cell+cell_width/2, y_cell-cell_height/2, cell, 
                       ha='center', va='center', fontsize=7, fontweight='bold')
            else:
                # 数据单元格
                if '↗' in cell:  # Zero-shot
                    bg_color = '#FFE6E6'
                elif '↘' in cell:  # Forgetting
                    bg_color = '#E6F3FF'
                elif i == j:  # 对角线
                    bg_color = '#E6FFE6'
                else:
                    bg_color = 'white'
                
                rect = Rectangle((x_cell, y_cell-cell_height), cell_width*0.95, cell_height*0.9,
                                facecolor=bg_color, edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)
                ax.text(x_cell+cell_width/2, y_cell-cell_height/2, cell, 
                       ha='center', va='center', fontsize=6)
    
    # 图例
    legend_items = [
        ('Diagonal: Current Performance', '#E6FFE6'),
        ('Above: Zero-Shot Transfer ↗', '#FFE6E6'),
        ('Below: Backward Transfer ↘', '#E6F3FF')
    ]
    
    y_legend = 0.3
    for i, (label, color) in enumerate(legend_items):
        rect = Rectangle((10+i*2.5, y_legend), 0.3, 0.15, facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(10.4+i*2.5, y_legend+0.075, label, ha='left', va='center', fontsize=6)
    
    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 持续学习流程图已保存到: {save_path}")
    plt.close()


def create_fusion_details(save_path='figures/fusion_details.png'):
    """
    图4: 多模态融合详细图
    展示不同的融合策略
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 创建3个子图
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 子图1: Concat Fusion
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    ax1.set_title('(a) Concat Fusion', fontsize=12, fontweight='bold')
    
    # 输入
    draw_box(ax1, (1, 6.5), 2, 0.6, 'Text\nH_t ∈ ℝ^(L×D)', COLORS['encoder'], fontsize=8)
    draw_box(ax1, (4, 6.5), 2, 0.6, 'Image\nH_i ∈ ℝ^D', COLORS['encoder'], fontsize=8)
    
    # 拼接
    draw_box(ax1, (2.5, 5.0), 3, 0.6, 'Concat: [H_t; H_i_expanded]', COLORS['fusion'], fontsize=8)
    draw_arrow(ax1, (2, 6.5), (3.5, 5.6))
    draw_arrow(ax1, (5, 6.5), (4.5, 5.6))
    
    # FC
    draw_box(ax1, (2.5, 3.5), 3, 0.6, 'FC(2D → D)', COLORS['head'], fontsize=8)
    draw_arrow(ax1, (4, 5.0), (4, 4.1))
    
    # 输出
    draw_box(ax1, (2.5, 2.0), 3, 0.6, 'Fused\nF ∈ ℝ^(L×D)', COLORS['output'], fontsize=8)
    draw_arrow(ax1, (4, 3.5), (4, 2.6))
    
    # 公式
    ax1.text(5, 1.0, 'F = FC([H_t; H_i])', ha='center', fontsize=9, style='italic', family='monospace')
    
    # 子图2: Gated Fusion  
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    ax2.set_title('(b) Gated Fusion (Proposed)', fontsize=12, fontweight='bold', color='red')
    
    # 输入
    draw_box(ax2, (1, 6.5), 2, 0.6, 'Text\nH_t', COLORS['encoder'], fontsize=8)
    draw_box(ax2, (4, 6.5), 2, 0.6, 'Image\nH_i', COLORS['encoder'], fontsize=8)
    
    # 门控
    draw_box(ax2, (0.5, 5.0), 2, 0.5, 'Gate_t\n= σ(W_t·H_t)', COLORS['label_emb'], fontsize=7)
    draw_box(ax2, (3.5, 5.0), 2, 0.5, 'Gate_i\n= σ(W_i·H_i)', COLORS['label_emb'], fontsize=7)
    
    draw_arrow(ax2, (2, 6.5), (1.5, 5.5))
    draw_arrow(ax2, (5, 6.5), (4.5, 5.5))
    
    # 归一化
    draw_box(ax2, (2, 4.0), 3, 0.4, 'Normalize: g_t, g_i = Gate_t, Gate_i / (Gate_t + Gate_i)', 
             '#FFF4E6', fontsize=7)
    draw_arrow(ax2, (1.5, 5.0), (2.8, 4.4))
    draw_arrow(ax2, (4.5, 5.0), (4.2, 4.4))
    
    # 加权融合
    draw_box(ax2, (2.5, 2.8), 3, 0.6, 'Weighted Fusion\nF = g_t ⊙ H_t + g_i ⊙ H_i', 
             COLORS['fusion'], fontsize=8, fontweight='bold')
    draw_arrow(ax2, (3.5, 4.0), (4, 3.4))
    
    # 输出
    draw_box(ax2, (2.5, 1.5), 3, 0.6, 'Adaptive\nFused Features', COLORS['output'], fontsize=8)
    draw_arrow(ax2, (4, 2.8), (4, 2.1))
    
    # 优势
    ax2.text(5, 0.8, '✓ Dynamic weights', ha='center', fontsize=7, color='darkgreen')
    ax2.text(5, 0.5, '✓ Task-adaptive', ha='center', fontsize=7, color='darkgreen')
    
    # 子图3: Attention Fusion
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 8)
    ax3.axis('off')
    ax3.set_title('(c) Attention Fusion (Proposed)', fontsize=12, fontweight='bold', color='red')
    
    # 输入
    draw_box(ax3, (1, 6.5), 2, 0.6, 'Text (Q)\nH_t', COLORS['encoder'], fontsize=8)
    draw_box(ax3, (4, 6.5), 2, 0.6, 'Image (K,V)\nH_i', COLORS['encoder'], fontsize=8)
    
    # Cross-attention
    draw_box(ax3, (2, 5.0), 4, 0.8, 'Multi-Head Cross-Attention\nAttn(Q, K, V)', 
             COLORS['fusion'], fontsize=8, fontweight='bold')
    draw_arrow(ax3, (2, 6.5), (2.5, 5.8))
    draw_arrow(ax3, (5, 6.5), (5.5, 5.8))
    
    ax3.text(4, 5.4, 'Attention(Q,K,V) = softmax(QK^T/√d)V', ha='center', fontsize=7, style='italic', family='monospace')
    
    # FFN
    draw_box(ax3, (2.5, 3.5), 3, 0.6, 'Feed-Forward Network', COLORS['head'], fontsize=8)
    draw_arrow(ax3, (4, 5.0), (4, 4.1))
    
    # 输出
    draw_box(ax3, (2.5, 2.0), 3, 0.6, 'Context-aware\nFused Features', COLORS['output'], fontsize=8)
    draw_arrow(ax3, (4, 3.5), (4, 2.6))
    
    # 公式
    ax3.text(5, 1.0, 'F = FFN(H_t + CrossAttn(H_t, H_i))', ha='center', fontsize=8, style='italic', family='monospace')
    
    # 子图4: Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 8)
    ax4.axis('off')
    ax4.set_title('(d) Fusion Strategy Comparison', fontsize=12, fontweight='bold')
    
    # 对比表格
    strategies = ['Concat', 'Add', 'Gated*', 'Attention*']
    metrics = {
        'Parameters': ['2D→D', '0', '2×(D²+D)', '12D²'],
        'Adaptivity': ['✗', '✗', '✓', '✓'],
        'Complexity': ['O(LD)', 'O(LD)', 'O(LD)', 'O(L²D)']
    }
    
    # 表头
    ax4.text(1, 7.0, 'Strategy', ha='left', fontsize=9, fontweight='bold')
    ax4.text(3, 7.0, 'Parameters', ha='center', fontsize=9, fontweight='bold')
    ax4.text(5.5, 7.0, 'Adaptivity', ha='center', fontsize=9, fontweight='bold')
    ax4.text(8, 7.0, 'Complexity', ha='center', fontsize=9, fontweight='bold')
    
    y_table = 6.5
    for i, strategy in enumerate(strategies):
        y = y_table - i * 0.5
        
        # 策略名（带*表示新增）
        is_new = '*' in strategy
        color = 'red' if is_new else 'black'
        weight = 'bold' if is_new else 'normal'
        ax4.text(1, y, strategy, ha='left', fontsize=8, color=color, fontweight=weight)
        
        # 指标
        ax4.text(3, y, metrics['Parameters'][i], ha='center', fontsize=7, family='monospace')
        ax4.text(5.5, y, metrics['Adaptivity'][i], ha='center', fontsize=8)
        ax4.text(8, y, metrics['Complexity'][i], ha='center', fontsize=7, family='monospace')
    
    # 注释
    ax4.text(5, 3.5, '* Proposed in this work', ha='center', fontsize=8, color='red', style='italic')
    
    # 性能示意图
    ax4.text(2, 2.5, 'Empirical Performance:', ha='left', fontsize=9, fontweight='bold')
    
    # 简单的柱状图
    perf_data = [85.2, 83.1, 88.5, 87.3]
    x_bars = np.arange(len(strategies))
    bar_width = 0.6
    
    for i, (strat, perf) in enumerate(zip(strategies, perf_data)):
        x = 1.5 + i * 1.5
        height = perf / 20  # 缩放
        color = '#FF9999' if '*' in strat else '#9999FF'
        rect = Rectangle((x, 1.0), 0.4, height, facecolor=color, edgecolor='black', linewidth=1)
        ax4.add_patch(rect)
        ax4.text(x+0.2, 1.0+height+0.1, f'{perf:.1f}%', ha='center', fontsize=6)
    
    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 多模态融合详细图已保存到: {save_path}")
    plt.close()


def create_training_algorithm(save_path='figures/training_algorithm.png'):
    """
    图5: 训练算法伪代码
    适合放在论文中
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(6, 9.5, 'Algorithm: Label Embedding Guided Continual Learning',
            ha='center', fontsize=14, fontweight='bold')
    
    # 算法框
    algo_box = mpatches.FancyBboxPatch(
        (0.5, 0.5), 11, 8.5,
        boxstyle="round,pad=0.2",
        edgecolor='black',
        facecolor='#FAFAFA',
        linewidth=2
    )
    ax.add_patch(algo_box)
    
    # 伪代码
    algorithm = [
        ('Input:', 'Task sequence T = {T₁, T₂, ..., T_K}, datasets D = {D₁, ..., D_K}', 'bold', 10),
        ('Output:', 'Model θ_K that performs well on all tasks', 'bold', 10),
        ('', '', 'normal', 8),
        ('1:', 'Initialize:', 'bold', 10),
        ('', '  • Text encoder φ_t, Image encoder φ_i', 'normal', 9),
        ('', '  • Global label embedding E_label ∈ ℝ^(N×d)', 'normal', 9),
        ('', '  • CL components: F_EWC = ∅, M_replay = ∅, ...', 'normal', 9),
        ('', '', 'normal', 8),
        ('2:', 'for task t = 1 to K do:', 'bold', 10),
        ('', '  • Load data D_t and previous model θ_(t-1)', 'normal', 9),
        ('', '  • Freeze old label embeddings: E_label[old] ← frozen', 'normal', 9),
        ('', '  • Create/load task head h_t', 'normal', 9),
        ('', '', 'normal', 8),
        ('', '  for epoch = 1 to E do:', 'bold', 9),
        ('', '    for batch (x_text, x_img, y) in D_t do:', 'normal', 9),
        ('', '      // Encoding', 'normal', 8),
        ('', '      H_t ← φ_t(x_text), H_i ← φ_i(x_img)', 'normal', 8),
        ('', '      ', 'normal', 8),
        ('', '      // Fusion', 'normal', 8),
        ('', '      F ← Fusion(H_t, H_i)  // Gated/Attention/Concat', 'normal', 8),
        ('', '      ', 'normal', 8),
        ('', '      // Label-Attentive Prediction', 'normal', 8),
        ('', '      z_labels ← E_label[task_t]  // Get task-specific embeddings', 'normal', 8),
        ('', '      logits ← TokenProj(F) · LabelProj(z_labels)^T / √d', 'normal', 8),
        ('', '      ', 'normal', 8),
        ('', '      // Loss Computation', 'normal', 8),
        ('', '      L_CE ← CrossEntropy(logits, y)', 'normal', 8),
        ('', '      L_CL ← Σ_strategy CL_loss(θ, θ_old)  // EWC/Replay/etc', 'normal', 8),
        ('', '      L_sim ← Σ_groups ||cos_sim(e_i, e_j) - 1||²', 'normal', 8),
        ('', '      L ← L_CE + λ_CL·L_CL + λ_sim·L_sim', 'normal', 8),
        ('', '      ', 'normal', 8),
        ('', '      // Update', 'normal', 8),
        ('', '      θ ← θ - α∇_θ L  // Only update shared params & new labels', 'normal', 8),
        ('', '    end for', 'normal', 9),
        ('', '  end for', 'bold', 9),
        ('', '', 'normal', 8),
        ('', '  // Update CL Components', 'normal', 9),
        ('', '  if EWC: F_EWC ← F_EWC ∪ {Fisher(θ, D_t)}', 'normal', 9),
        ('', '  if Replay: M_replay ← M_replay ∪ {Sample(D_t)}', 'normal', 9),
        ('', '  ', 'normal', 8),
        ('', '  // Zero-Shot Evaluation (Optional)', 'normal', 9),
        ('', '  for task t\' in {t+1, ..., K} do:', 'normal', 9),
        ('', '    Evaluate θ_t on D_t\' using current label embeddings', 'normal', 8),
        ('', '  end for', 'normal', 9),
        ('', '', 'normal', 8),
        ('3:', 'return θ_K', 'bold', 10),
    ]
    
    y_pos = 8.5
    for line_num, content, weight, size in algorithm:
        if line_num:
            # 行号
            ax.text(1.0, y_pos, line_num, ha='left', fontsize=size, fontweight=weight, family='monospace')
            ax.text(1.5, y_pos, content, ha='left', fontsize=size, fontweight=weight, family='monospace')
        else:
            # 缩进内容
            ax.text(1.5, y_pos, content, ha='left', fontsize=size, fontweight=weight, family='monospace')
        
        y_pos -= 0.18
    
    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练算法伪代码已保存到: {save_path}")
    plt.close()


def create_task_sequence_diagram(save_path='figures/task_sequence.png'):
    """
    图6: 任务序列和模态切换图
    展示text-to-multimodal学习范式
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 标题
    ax.text(7, 5.5, 'Task Sequence: Text-to-Multimodal Learning Paradigm',
            ha='center', fontsize=14, fontweight='bold')
    
    # 阶段1: Text-Only
    stage1_box = mpatches.FancyBboxPatch(
        (0.5, 3.5), 6, 1.5,
        boxstyle="round,pad=0.1",
        edgecolor='green',
        facecolor='#F0FFF0',
        linewidth=2
    )
    ax.add_patch(stage1_box)
    ax.text(3.5, 4.8, 'Stage 1: Text-Only Learning (t=1~4)', ha='center', fontsize=11, fontweight='bold', color='darkgreen')
    
    tasks_stage1 = ['MASC', 'MATE', 'MNER', 'MABSA']
    for i, task in enumerate(tasks_stage1):
        x = 0.8 + i * 1.45
        draw_box(ax, (x, 3.7), 1.3, 0.5, f'T{i+1}: {task}\n📝 Text', '#90EE90', fontsize=8)
    
    # 阶段2: Multimodal
    stage2_box = mpatches.FancyBboxPatch(
        (7.5, 3.5), 6, 1.5,
        boxstyle="round,pad=0.1",
        edgecolor='blue',
        facecolor='#F0F8FF',
        linewidth=2
    )
    ax.add_patch(stage2_box)
    ax.text(10.5, 4.8, 'Stage 2: Multimodal Learning (t=5~8)', ha='center', fontsize=11, fontweight='bold', color='darkblue')
    
    tasks_stage2 = ['MASC', 'MATE', 'MNER', 'MABSA']
    for i, task in enumerate(tasks_stage2):
        x = 7.8 + i * 1.45
        draw_box(ax, (x, 3.7), 1.3, 0.5, f'T{i+5}: {task}\n📝🖼 Text+Img', '#87CEEB', fontsize=8)
    
    # 转移箭头
    draw_arrow(ax, (6.5, 4.5), (7.5, 4.5), style='->', linewidth=3, color='red')
    ax.text(7, 4.7, 'Modal Shift', ha='center', fontsize=9, color='red', fontweight='bold')
    
    # ========== 知识传递示意 ==========
    y_transfer = 2.5
    
    # Text知识
    draw_box(ax, (1.5, y_transfer), 3.5, 0.6, 'Text Knowledge: Linguistic Patterns', '#E6F7E6', fontsize=8)
    
    # Multimodal知识
    draw_box(ax, (8.5, y_transfer), 4.0, 0.6, 'Multimodal Knowledge: Vision-Language Alignment', '#E6F0FF', fontsize=8)
    
    # 共享Label Embedding
    draw_box(ax, (4.5, y_transfer-1.2), 5.0, 0.7, 'Shared Label Embeddings E_label\n(Bridge for Knowledge Transfer)', 
             COLORS['label_emb'], fontsize=9, fontweight='bold')
    
    # 箭头
    draw_arrow(ax, (3.0, y_transfer), (5.5, y_transfer-1.2), linewidth=2, color='purple')
    draw_arrow(ax, (10.0, y_transfer), (8.5, y_transfer-1.2), linewidth=2, color='purple')
    
    # 优势标注
    advantages = [
        '✓ Forward Transfer: Text → Multimodal',
        '✓ Backward Transfer: Multimodal → Text',
        '✓ Zero-Shot: Predict unseen tasks'
    ]
    
    y_adv = 0.5
    for i, adv in enumerate(advantages):
        ax.text(1, y_adv-i*0.2, adv, ha='left', fontsize=8, color='darkgreen')
    
    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 任务序列图已保存到: {save_path}")
    plt.close()


def create_all_figures():
    """生成所有图表"""
    print("\n" + "="*80)
    print("生成科研论文级别的架构图")
    print("="*80)
    
    # 创建输出目录
    Path('figures').mkdir(exist_ok=True)
    
    print("\n正在生成图表...")
    
    try:
        # 图1: 整体架构
        print("\n[1/5] 生成整体架构图...")
        create_overall_architecture()
        
        # 图2: Label Embedding详细图
        print("[2/5] 生成Label Embedding详细图...")
        create_label_embedding_details()
        
        # 图3: 持续学习流程
        print("[3/5] 生成持续学习流程图...")
        create_continual_learning_flow()
        
        # 图4: 融合详细图
        print("[4/5] 生成多模态融合详细图...")
        create_fusion_details()
        
        # 图5: 训练算法
        print("[5/5] 生成训练算法伪代码...")
        create_training_algorithm()
        
        # 图6: 任务序列
        print("[6/6] 生成任务序列图...")
        create_task_sequence_diagram()
        
        print("\n" + "="*80)
        print("✅ 所有图表生成完成！")
        print("="*80)
        
        print("\n生成的图表:")
        figures = [
            'overall_architecture.png - 整体架构图（用于Introduction/Method）',
            'label_embedding_details.png - Label Embedding机制（核心创新）',
            'continual_learning_flow.png - 持续学习流程（完整过程）',
            'fusion_details.png - 多模态融合对比（方法对比）',
            'training_algorithm.png - 训练算法伪代码（Algorithm部分）',
            'task_sequence.png - 任务序列和知识传递（Experimental Setup）'
        ]
        
        for i, desc in enumerate(figures, 1):
            print(f"  {i}. figures/{desc}")
        
        print("\n💡 使用建议:")
        print("  • Figure 1: 放在Introduction或Method Overview")
        print("  • Figure 2: 放在Method部分，详细说明Label Embedding")
        print("  • Figure 3: 放在Method部分，说明训练流程")
        print("  • Figure 4: 放在Ablation Study，对比不同融合策略")
        print("  • Figure 5: 放在Algorithm框中")
        print("  • Figure 6: 放在Experimental Setup")
        
        print("\n📝 LaTeX引用示例:")
        print("  \\begin{figure}[t]")
        print("    \\centering")
        print("    \\includegraphics[width=\\textwidth]{figures/overall_architecture.png}")
        print("    \\caption{Overall architecture of our proposed method.}")
        print("    \\label{fig:architecture}")
        print("  \\end{figure}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = create_all_figures()
    sys.exit(0 if success else 1)

