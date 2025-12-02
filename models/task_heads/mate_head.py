# models/task_heads/mate_head.py
from models.task_heads.mner_head import MNERHead

class MATEHead(MNERHead):
    """
    MATE任务头 (无 BiLSTM) - 继承自 MNERHead
    
    架构：
        Input (768) 
        → Dropout 
        → Linear (3) 
        → CRF (可选)
        
    说明：MATE (Aspect Term Extraction) 是一个 3 分类序列标注任务 (O, B, I)
    """
    
    def __init__(
        self, 
        input_dim: int = 768, 
        num_labels: int = 3,       # MATE 默认为 3 (O, B, I)
        dropout_prob: float = 0.3, # Transformer 常用 dropout
        hidden_dim: int = None,    # 兼容参数，无实际作用
        use_crf: bool = True
    ):
        # 显式调用父类构造函数，传递 MATE 特定的参数
        super().__init__(
            input_dim=input_dim,
            num_labels=num_labels,
            dropout_prob=dropout_prob,
            hidden_dim=hidden_dim,
            use_crf=use_crf
        )