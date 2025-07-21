import torch
import torch.nn as nn
from models.task_heads.sent_label_attn import LabelAttentionSentHead
from models.task_heads.token_label_heads import TokenLabelHead
from continual.label_embedding import GlobalLabelEmbedding, build_global_label_mapping
from modules.train_utils import Full_Model
from models.base_model import BaseMultimodalModel

def test_full_training_flow():
    print("=== 测试完整训练流程：多任务独立头管理 ===")
    
    # 创建标签嵌入
    label2idx = build_global_label_mapping()
    label_emb = GlobalLabelEmbedding(label2idx, emb_dim=128)
    
    # 创建基础模型 - 使用本地模型避免网络问题
    try:
        base_model = BaseMultimodalModel(
            text_model_name="downloaded_model/deberta-v3-base",
            image_model_name="downloaded_model/vit-base-patch16-224-in21k",
            multimodal_fusion="concat",
            num_heads=8,
            mode="text_only"
        )
    except:
        # 如果本地模型不存在，创建一个模拟的基础模型
        print("本地模型不存在，创建模拟基础模型...")
        class MockBaseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_hidden_size = 768
                self.mode = "text_only"
            
            def forward(self, input_ids, attention_mask, token_type_ids, image_tensor, return_sequence=False):
                batch_size = input_ids.size(0)
                if return_sequence:
                    return torch.randn(batch_size, input_ids.size(1), 768)
                else:
                    return torch.randn(batch_size, 768)
        
        base_model = MockBaseModel()
    
    # 模拟训练流程：第一个任务 MASC
    print("\n--- 训练第一个任务：MASC ---")
    masc_head = LabelAttentionSentHead(
        input_dim=768,
        num_labels=3,
        label_emb=label_emb,
        task_name="masc"
    )
    
    full_model = Full_Model(base_model, masc_head, dropout_prob=0.1)
    full_model.add_task_head("session_1_masc", "masc", masc_head, None)
    full_model.set_active_head("session_1_masc")
    
    print(f"当前活动头: {type(full_model.head).__name__}")
    print(f"当前任务: {full_model.get_current_task_name()}")
    
    # 模拟训练第二个任务：MABSA
    print("\n--- 训练第二个任务：MABSA ---")
    mabsa_head = TokenLabelHead(
        input_dim=768,
        hidden_dim=256,
        num_labels=7,
        label_emb=label_emb,
        task_name="mabsa"
    )
    
    # 添加新的任务头（不覆盖之前的）
    full_model.add_task_head("session_2_mabsa", "mabsa", mabsa_head, None)
    full_model.set_active_head("session_2_mabsa")
    
    print(f"当前活动头: {type(full_model.head).__name__}")
    print(f"当前任务: {full_model.get_current_task_name()}")
    print(f"总任务头数量: {len(full_model.task_heads)}")
    
    # 模拟训练第三个任务：MATE
    print("\n--- 训练第三个任务：MATE ---")
    mate_head = TokenLabelHead(
        input_dim=768,
        hidden_dim=256,
        num_labels=3,  # MATE 只有3个标签
        label_emb=label_emb,
        task_name="mate"
    )
    
    # 添加新的任务头
    full_model.add_task_head("session_3_mate", "mate", mate_head, None)
    full_model.set_active_head("session_3_mate")
    
    print(f"当前活动头: {type(full_model.head).__name__}")
    print(f"当前任务: {full_model.get_current_task_name()}")
    print(f"总任务头数量: {len(full_model.task_heads)}")
    
    # 测试切换回之前的任务
    print("\n--- 测试任务切换 ---")
    print("切换到 MASC 任务:")
    full_model.set_active_head("session_1_masc")
    print(f"  活动头: {type(full_model.head).__name__}")
    print(f"  任务: {full_model.get_current_task_name()}")
    
    print("切换到 MABSA 任务:")
    full_model.set_active_head("session_2_mabsa")
    print(f"  活动头: {type(full_model.head).__name__}")
    print(f"  任务: {full_model.get_current_task_name()}")
    
    print("切换到 MATE 任务:")
    full_model.set_active_head("session_3_mate")
    print(f"  活动头: {type(full_model.head).__name__}")
    print(f"  任务: {full_model.get_current_task_name()}")
    
    # 验证每个任务头的参数是独立的
    print("\n--- 验证任务头独立性 ---")
    masc_params = full_model.task_heads["session_1_masc"]["head"].state_dict()
    mabsa_params = full_model.task_heads["session_2_mabsa"]["head"].state_dict()
    mate_params = full_model.task_heads["session_3_mate"]["head"].state_dict()
    
    print(f"MASC 头参数数量: {len(masc_params)}")
    print(f"MABSA 头参数数量: {len(mabsa_params)}")
    print(f"MATE 头参数数量: {len(mate_params)}")
    
    # 检查关键参数形状
    if "sent_projection.weight" in masc_params:
        print(f"MASC sent_projection.weight 形状: {masc_params['sent_projection.weight'].shape}")
    
    if "U" in mabsa_params:
        print(f"MABSA U 形状: {mabsa_params['U'].shape}")
    
    if "U" in mate_params:
        print(f"MATE U 形状: {mate_params['U'].shape}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_full_training_flow() 