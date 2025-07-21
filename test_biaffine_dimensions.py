import torch
import torch.nn as nn
from continual.label_embedding import GlobalLabelEmbedding, build_global_label_mapping
from models.task_heads.token_label_heads import TokenLabelHead

def test_token_label_head_dimensions():
    print("=== 测试 TokenLabelHead 维度匹配 ===")
    
    # 创建标签嵌入
    label2idx = build_global_label_mapping()
    label_emb = GlobalLabelEmbedding(label2idx, emb_dim=128)
    
    # 创建模型
    input_dim = 768  # BERT 隐藏维度
    hidden_dim = 256  # 隐藏维度
    num_labels = 7   # MABSA 任务有7个标签
    task_name = "mabsa"
    
    model = TokenLabelHead(input_dim, hidden_dim, num_labels, label_emb=label_emb, task_name=task_name)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 10
    seq_feats = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"输入序列特征形状: {seq_feats.shape}")
    print(f"隐藏维度: {hidden_dim}")
    print(f"标签数量: {num_labels}")
    output = model(seq_feats)
    print(f"输出形状: {output.shape}")
    assert output.shape == (batch_size, seq_len, num_labels), f"输出形状不匹配: {output.shape}"

if __name__ == "__main__":
    test_token_label_head_dimensions() 