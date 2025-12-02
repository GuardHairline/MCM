# models/task_heads/mate_head_bilstm.py
from models.task_heads.mner_head_bilstm import MNERHeadBiLSTM

class MATEHeadBiLSTM(MNERHeadBiLSTM):
    """
    MATE任务头 - BiLSTM增强版
    
    架构：BiLSTM + CRF
    默认标签数：3 (O, B, I)
    """
    
    def __init__(
        self, 
        input_dim: int = 768, 
        num_labels: int = 3, # MATE 默认 3
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        use_crf: bool = True,
    ):
        # 显式传递参数，确保 defaults 不会被父类覆盖
        super().__init__(
            input_dim=input_dim,
            num_labels=num_labels,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            use_crf=use_crf,
        )