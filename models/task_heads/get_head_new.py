from models.task_heads.biaffine_heads import BiaffineSpanHead
from models.task_heads.sent_label_attn import LabelAttentionSentHead
from models.task_heads.token_label_heads import TokenLabelHead
from continual.label_embedding import GlobalLabelEmbedding
from continual.label_embedding_manager import LabelEmbeddingManager

def get_head(task, base_model, args, label_emb: GlobalLabelEmbedding = None):
    """
    获取任务头部，支持标签嵌入
    
    Args:
        task: 任务名称
        base_model: 基础模型
        args: 参数
        label_emb: 标签嵌入管理器
    """
    # 获取任务标签数量
    if label_emb is not None:
        num_labels = label_emb.get_task_num_labels(task)
    else:
        # 如果没有标签嵌入，使用默认的标签数量
        task_num_labels = {
            "mabsa": 7,   # O, B-NEG, I-NEG, B-NEU, I-NEU, B-POS, I-POS
            "masc": 3,    # NEG, NEU, POS
            "mate": 3,    # O, B, I
            "mner": 9     # O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
        }
        num_labels = task_num_labels.get(task, 3)
    
    if task in ["mate", "mner", "mabsa"]:           # token-level
        return TokenLabelHead(
            input_dim=base_model.fusion_output_dim,
            hidden_dim=getattr(args, 'span_hidden', 256),
            num_labels=num_labels,
            label_emb=label_emb,
            task_name=task,
            use_crf=(args.num_labels > 3),
        )
    elif task == "masc":                             # sentence-level
        return LabelAttentionSentHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=num_labels,
            label_emb=label_emb,
            task_name=task
        )
    else:
        raise ValueError(f"Unknown task: {task}")
