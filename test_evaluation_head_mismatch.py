import torch
import torch.nn as nn
from models.task_heads.sent_label_attn import LabelAttentionSentHead
from models.task_heads.token_label_heads import TokenLabelHead
from continual.label_embedding import GlobalLabelEmbedding, build_global_label_mapping

def test_head_type_check():
    print("=== 测试模型头类型检查 ===")
    
    # 创建标签嵌入
    label2idx = build_global_label_mapping()
    label_emb = GlobalLabelEmbedding(label2idx, emb_dim=128)
    
    # 测试 MASC 任务应该使用 LabelAttentionSentHead
    masc_head = LabelAttentionSentHead(
        input_dim=768,
        num_labels=3,
        label_emb=label_emb,
        task_name="masc"
    )
    
    print(f"MASC 任务模型头类型: {type(masc_head).__name__}")
    print(f"期望类型: LabelAttentionSentHead")
    print(f"匹配: {type(masc_head).__name__ == 'LabelAttentionSentHead'}")
    
    # 测试 MABSA 任务应该使用 TokenLabelHead
    mabsa_head = TokenLabelHead(
        input_dim=768,
        hidden_dim=256,
        num_labels=7,
        label_emb=label_emb,
        task_name="mabsa"
    )
    
    print(f"MABSA 任务模型头类型: {type(mabsa_head).__name__}")
    print(f"期望类型: TokenLabelHead")
    print(f"匹配: {type(mabsa_head).__name__ == 'TokenLabelHead'}")
    
    # 测试错误的情况
    print("\n=== 测试错误情况 ===")
    
    # 模拟 MASC 任务使用了错误的模型头
    wrong_head = TokenLabelHead(
        input_dim=768,
        hidden_dim=256,
        num_labels=3,
        label_emb=label_emb,
        task_name="masc"
    )
    
    current_head_type = type(wrong_head).__name__
    expected_head_type = "LabelAttentionSentHead"
    
    print(f"当前模型头类型: {current_head_type}")
    print(f"期望模型头类型: {expected_head_type}")
    print(f"是否匹配: {current_head_type == expected_head_type}")
    
    if current_head_type != expected_head_type:
        print("❌ 模型头类型不匹配！")
    else:
        print("✅ 模型头类型匹配！")

if __name__ == "__main__":
    test_head_type_check() 