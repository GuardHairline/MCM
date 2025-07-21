import torch
import torch.nn as nn
from models.task_heads.sent_label_attn import LabelAttentionSentHead
from models.task_heads.token_label_heads import TokenLabelHead
from continual.label_embedding import GlobalLabelEmbedding, build_global_label_mapping
from modules.train_utils import Full_Model
from models.base_model import BaseMultimodalModel

def test_model_parameter_compatibility():
    print("=== 测试模型参数兼容性检查 ===")
    
    # 创建标签嵌入
    label2idx = build_global_label_mapping()
    label_emb = GlobalLabelEmbedding(label2idx, emb_dim=128)
    
    # 创建基础模型
    base_model = BaseMultimodalModel(
        text_model_name="microsoft/deberta-v3-base",
        image_model_name="google/vit-base-patch16-224-in21k",
        multimodal_fusion="concat",
        num_heads=8,
        mode="text_only"
    )
    
    # 创建 MABSA 任务的模型头（7个标签）
    mabsa_head = TokenLabelHead(
        input_dim=768,
        hidden_dim=256,
        num_labels=7,
        label_emb=label_emb,
        task_name="mabsa"
    )
    
    # 创建 MASC 任务的模型头（3个标签）
    masc_head = LabelAttentionSentHead(
        input_dim=768,
        num_labels=3,
        label_emb=label_emb,
        task_name="masc"
    )
    
    # 创建 Full_Model，使用 MABSA 头
    full_model = Full_Model(base_model, mabsa_head, dropout_prob=0.1)
    
    # 模拟从检查点加载的参数（MABSA 任务的参数）
    checkpoint_params = {}
    for name, param in mabsa_head.named_parameters():
        checkpoint_params[f'head.{name}'] = param.data.clone()
    
    print(f"检查点参数数量: {len(checkpoint_params)}")
    print("检查点参数形状:")
    for key, value in checkpoint_params.items():
        print(f"  {key}: {value.shape}")
    
    # 现在切换到 MASC 头
    full_model.head = masc_head
    
    print(f"\n当前模型头类型: {type(full_model.head).__name__}")
    print("当前模型头参数形状:")
    for name, param in masc_head.named_parameters():
        print(f"  {name}: {param.shape}")
    
    # 测试参数兼容性检查
    print("\n--- 测试参数兼容性检查 ---")
    filtered_params = {}
    for key, value in checkpoint_params.items():
        if key.startswith('head.'):
            # 检查参数是否与当前任务兼容
            current_head = full_model.head
            param_name = key.replace('head.', '')
            if hasattr(current_head, param_name):
                current_param = getattr(current_head, param_name)
                if hasattr(current_param, 'shape') and current_param.shape == value.shape:
                    filtered_params[key] = value
                    print(f"✓ 兼容参数: {key} - 形状 {value.shape}")
                else:
                    print(f"✗ 不兼容参数: {key} - 检查点形状 {value.shape}, 当前形状 {current_param.shape}")
            else:
                print(f"✗ 参数不存在: {key}")
        else:
            filtered_params[key] = value
    
    print(f"\n过滤后的参数数量: {len(filtered_params)}")
    print("=== 测试完成 ===")

if __name__ == "__main__":
    test_model_parameter_compatibility() 