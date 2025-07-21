import torch
import torch.nn as nn
from continual.label_embedding import GlobalLabelEmbedding, build_global_label_mapping
from models.task_heads.sent_label_attn import LabelAttentionSentHead

def test_sent_label_attn():
    print("=== 测试 LabelAttentionSentHead 维度匹配 ===")
    
    # 创建标签嵌入
    label2idx = build_global_label_mapping()
    label_emb = GlobalLabelEmbedding(label2idx, emb_dim=128)
    
    # 创建模型
    input_dim = 768  # BERT 隐藏维度
    num_labels = 3   # MASC 任务有3个标签
    task_name = "masc"
    
    model = LabelAttentionSentHead(input_dim, num_labels, label_emb, task_name)
    
    # 创建测试输入
    batch_size = 4
    sent_vec = torch.randn(batch_size, input_dim)
    
    print(f"输入句子向量形状: {sent_vec.shape}")
    print(f"标签嵌入维度: {label_emb.emb_dim}")
    print(f"标签数量: {num_labels}")
    
    # 前向传播
    try:
        logits = model(sent_vec)
        print(f"输出 logits 形状: {logits.shape}")
        print(f"期望形状: ({batch_size}, {num_labels})")
        
        if logits.shape == (batch_size, num_labels):
            print("✅ 维度匹配正确！")
        else:
            print("❌ 维度不匹配！")
            
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_sent_label_attn() 