# \models\task_heads\mabsa_head.py
from models.task_heads.mner_head import MNERHead

class MABSAHead(MNERHead):
    """
    MABSA任务头 (无 BiLSTM) - 继承自 MNERHead
    
    架构：
        Input (768) 
        → Dropout 
        → Linear (7) 
        → CRF (可选)
        
    说明：MABSA (Multimodal Aspect-Based Sentiment Analysis) 是一个 7 分类序列标注任务
    (O + {NEG, NEU, POS} * {B, I})
    """
    
    def __init__(
        self, 
        input_dim: int = 768, 
        num_labels: int = 7,       # MABSA 默认为 7
        dropout_prob: float = 0.3, 
        hidden_dim: int = None,    
        use_crf: bool = True
    ):
        super().__init__(
            input_dim=input_dim,
            num_labels=num_labels,
            dropout_prob=dropout_prob,
            hidden_dim=hidden_dim,
            use_crf=use_crf
        )