#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用Graphviz生成专业的架构图
更适合科研论文发表

需要安装: pip install graphviz
"""

import sys
from pathlib import Path

try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    print("⚠️  Graphviz未安装，请运行: pip install graphviz")
    print("   或使用matplotlib版本: python visualize_architecture.py")
    GRAPHVIZ_AVAILABLE = False


def create_overall_architecture_graph(save_path='figures/architecture_graph'):
    """
    使用Graphviz创建整体架构图
    """
    if not GRAPHVIZ_AVAILABLE:
        return False
    
    dot = Digraph(comment='Label Embedding Guided Continual Learning Architecture')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.8')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # ========== 输入层 ==========
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Layer', style='dashed', color='blue')
        c.node('text_input', 'Text Input\n[CLS] w₁ w₂ ... [SEP]', fillcolor='#E8F4F8')
        c.node('image_input', 'Image Input\n224×224×3', fillcolor='#E8F4F8')
    
    # ========== 编码器层 ==========
    with dot.subgraph(name='cluster_encoders') as c:
        c.attr(label='Encoder Layer', style='dashed', color='darkgreen')
        c.node('text_encoder', 'Text Encoder\nDeBERTa-v3-base\n→ H_text ∈ ℝ^(L×768)', fillcolor='#B8E6F0')
        c.node('image_encoder', 'Image Encoder\nViT-base\n→ H_image ∈ ℝ^768', fillcolor='#B8E6F0')
    
    # ========== Label Embedding模块 ==========
    with dot.subgraph(name='cluster_label_emb') as c:
        c.attr(label='Label Embedding Module (Innovation)', style='bold', color='red')
        c.node('global_mapping', 'Global Label Mapping\n{(task, label_id) → idx}', fillcolor='#FFD9D9', shape='box')
        c.node('pretrained_emb', 'Pretrained Embeddings\n(DeBERTa-encoded)', fillcolor='#FFD9D9')
        c.node('label_groups', 'Semantic Label Groups\n{NEG, POS, ENTITY, ...}', fillcolor='#FFD9D9')
        c.node('label_embedding', 'Label Embedding Matrix\nE ∈ ℝ^(22×128)\n(Trainable)', 
               fillcolor='#FF9999', fontcolor='white', style='filled,bold')
        
        c.edge('global_mapping', 'label_embedding')
        c.edge('pretrained_emb', 'label_embedding', label='initialize')
        c.edge('label_groups', 'label_embedding', label='regularize')
    
    # ========== 融合层 ==========
    with dot.subgraph(name='cluster_fusion') as c:
        c.attr(label='Multimodal Fusion', style='dashed', color='orange')
        c.node('fusion', 'Adaptive Fusion\nF = φ(H_text, H_image)\nStrategies: Gated/Attention/Concat', 
               fillcolor='#FFE6CC', shape='box')
    
    # ========== 任务头 ==========
    with dot.subgraph(name='cluster_heads') as c:
        c.attr(label='Task-Specific Heads', style='dashed', color='purple')
        c.node('token_head', 'Token-Level Head\nLogits = TokenProj(F) · LabelProj(E)^T\n→ (B,L,C)', 
               fillcolor='#D9F0D9')
        c.node('sent_head', 'Sentence-Level Head\nLogits = SentProj(F) · LabelProj(E)^T\n→ (B,C)', 
               fillcolor='#D9F0D9')
    
    # ========== 持续学习组件 ==========
    with dot.subgraph(name='cluster_cl') as c:
        c.attr(label='Continual Learning Strategies', style='dashed', color='brown')
        c.node('ewc', 'EWC\nFisher×500', fillcolor='#FFF4CC', shape='ellipse')
        c.node('replay', 'Experience\nReplay', fillcolor='#FFF4CC', shape='ellipse')
        c.node('gem', 'GEM\nGradient Proj', fillcolor='#FFF4CC', shape='ellipse')
        c.node('others', 'LwF/SI/MAS\n...', fillcolor='#FFF4CC', shape='ellipse')
    
    # ========== 输出 ==========
    dot.node('output_token', 'Token Predictions\n(MATE/MNER/MABSA)', fillcolor='#F0E6FF', shape='box')
    dot.node('output_sent', 'Sentence Predictions\n(MASC)', fillcolor='#F0E6FF', shape='box')
    
    # ========== 连接 ==========
    # 输入到编码器
    dot.edge('text_input', 'text_encoder', label='tokenize')
    dot.edge('image_input', 'image_encoder', label='transform')
    
    # 编码器到融合
    dot.edge('text_encoder', 'fusion', label='H_text')
    dot.edge('image_encoder', 'fusion', label='H_image')
    
    # 融合到任务头
    dot.edge('fusion', 'token_head', label='F (seq)')
    dot.edge('fusion', 'sent_head', label='F (cls)')
    
    # Label embedding到任务头
    dot.edge('label_embedding', 'token_head', label='E_labels', color='red', style='bold')
    dot.edge('label_embedding', 'sent_head', label='E_labels', color='red', style='bold')
    
    # 任务头到输出
    dot.edge('token_head', 'output_token')
    dot.edge('sent_head', 'output_sent')
    
    # CL组件到融合（正则化）
    dot.edge('ewc', 'fusion', label='L_ewc', style='dashed', color='gray')
    dot.edge('replay', 'fusion', label='L_replay', style='dashed', color='gray')
    dot.edge('gem', 'fusion', label='grad_proj', style='dashed', color='gray')
    
    # 保存
    dot.render(save_path, format='png', cleanup=True)
    print(f"✓ Graphviz架构图已保存到: {save_path}.png")
    
    # 同时保存源文件（可编辑）
    with open(f'{save_path}.dot', 'w') as f:
        f.write(dot.source)
    print(f"✓ Graphviz源文件已保存到: {save_path}.dot")
    
    return True


def create_label_embedding_graph(save_path='figures/label_embedding_graph'):
    """
    Label Embedding详细流程图
    """
    if not GRAPHVIZ_AVAILABLE:
        return False
    
    dot = Digraph(comment='Label Embedding Mechanism')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='filled', fontname='Arial')
    
    # ========== Label Definition ==========
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Label Definitions', rank='same')
        c.node('label_mabsa', 'MABSA Labels\n{O, B-NEG, I-NEG, B-NEU,\nI-NEU, B-POS, I-POS}', fillcolor='#FFE6CC')
        c.node('label_mate', 'MATE Labels\n{O, B, I}', fillcolor='#FFE6CC')
        c.node('label_mner', 'MNER Labels\n{O, B-PER, I-PER,\nB-ORG, ...}', fillcolor='#FFE6CC')
        c.node('label_masc', 'MASC Labels\n{NEG, NEU, POS}', fillcolor='#FFE6CC')
    
    # ========== Global Mapping ==========
    dot.node('global_map', 'Global Mapping\nlabel2idx:\n(MABSA,0)→0, (MABSA,1)→1, ...\n(MATE,0)→7, (MATE,1)→8, ...\nTotal: 22 global labels', 
             fillcolor='#FFFACD', shape='box')
    
    # ========== Label Text Descriptions ==========
    dot.node('label_texts', 'Label Text Descriptions\n(MABSA,0): "outside"\n(MABSA,1): "begin negative aspect"\n(MASC,0): "negative sentiment"\n...', 
             fillcolor='#E6E6FA', shape='note')
    
    # ========== DeBERTa Encoding ==========
    dot.node('deberta', 'DeBERTa-v3-base\nEncode label texts\n→ Pretrained Embeddings', 
             fillcolor='#B8E6F0', shape='box')
    
    # ========== Learnable Embedding ==========
    dot.node('embedding_matrix', 'nn.Embedding(22, 128)\nInitialized with pretrained\nTrainable for new tasks\nFrozen for old tasks', 
             fillcolor='#FF9999', fontcolor='white', style='bold,filled')
    
    # ========== Semantic Groups ==========
    dot.node('semantic_groups', 'Semantic Grouping\nNEG: {(MABSA,1), (MABSA,2), (MASC,0)}\nPOS: {(MABSA,5), (MABSA,6), (MASC,2)}\nENTITY: {(MATE,1), (MNER,1), ...}', 
             fillcolor='#FFD9D9', shape='folder')
    
    # ========== Similarity Regularization ==========
    dot.node('similarity_reg', 'Similarity Regularization\nL_sim = Σ_groups Σ_(i,j)∈group ||cos_sim(e_i, e_j) - 1||²\nEncourages semantic coherence', 
             fillcolor='#FFF4CC', shape='box')
    
    # ========== 连接 ==========
    dot.edge('label_mabsa', 'global_map')
    dot.edge('label_mate', 'global_map')
    dot.edge('label_mner', 'global_map')
    dot.edge('label_masc', 'global_map')
    
    dot.edge('global_map', 'label_texts', label='generate')
    dot.edge('label_texts', 'deberta', label='encode')
    dot.edge('deberta', 'embedding_matrix', label='initialize')
    
    dot.edge('global_map', 'semantic_groups', label='group by\nsemantics')
    dot.edge('semantic_groups', 'similarity_reg')
    dot.edge('similarity_reg', 'embedding_matrix', label='regularize', style='dashed', color='red')
    
    # 保存
    dot.render(save_path, format='png', cleanup=True)
    print(f"✓ Label Embedding流程图已保存到: {save_path}.png")
    
    return True


def main():
    """主函数"""
    print("\n" + "="*80)
    print("科研论文级别架构图生成工具")
    print("="*80)
    
    print("\n选项:")
    print("  1. 使用Matplotlib生成（无需额外依赖）")
    print("  2. 使用Graphviz生成（需要安装graphviz）")
    print("  3. 同时生成两种格式（推荐）")
    
    choice = input("\n请选择 (1/2/3，默认3): ").strip() or '3'
    
    success = True
    
    if choice in ['1', '3']:
        print("\n使用Matplotlib生成...")
        success = success and create_all_figures()
    
    if choice in ['2', '3']:
        if GRAPHVIZ_AVAILABLE:
            print("\n使用Graphviz生成...")
            success = success and create_overall_architecture_graph()
            success = success and create_label_embedding_graph()
        else:
            print("\n⚠️  Graphviz未安装，跳过")
            if choice == '2':
                success = False
    
    if success:
        print("\n" + "="*80)
        print("✅ 所有图表生成成功！")
        print("="*80)
        print("\n📁 输出目录: ./figures/")
        print("\n🎓 适用场景:")
        print("  • AAAI/CVPR/ACL等顶会投稿")
        print("  • 高质量期刊论文")
        print("  • 学术报告和展示")
        print("\n💡 LaTeX使用提示:")
        print("  \\usepackage{graphicx}")
        print("  \\includegraphics[width=\\columnwidth]{figures/overall_architecture.png}")
        return 0
    else:
        print("\n❌ 部分图表生成失败")
        return 1


if __name__ == "__main__":
    # 直接生成matplotlib版本（不需要交互）
    success = create_all_figures()
    
    # 尝试生成graphviz版本
    if GRAPHVIZ_AVAILABLE:
        print("\n同时生成Graphviz版本...")
        success = success and create_overall_architecture_graph()
        success = success and create_label_embedding_graph()
    
    sys.exit(0 if success else 1)

