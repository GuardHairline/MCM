# models/task_heads/mabsa_head_bilstm.py
from models.task_heads.mner_head_bilstm import MNERHeadBiLSTM

class MABSAHeadBiLSTM(MNERHeadBiLSTM):
    """
    MABSA任务头 - BiLSTM增强版
    
    架构：BiLSTM + CRF
    默认标签数：7 (O + 3情感*2)
    """
    
    def __init__(
        self, 
        input_dim: int = 768, 
        num_labels: int = 7, # MABSA 默认 7
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        use_crf: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            num_labels=num_labels,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            use_crf=use_crf,
        )