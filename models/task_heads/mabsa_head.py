from models.task_heads.mabsa_head_bilstm import MABSAHeadBiLSTM


class MABSAHead(MABSAHeadBiLSTM):
    """
    兼容旧接口的MABSA任务头，复用BiLSTM实现但禁用BiLSTM部分。
    """

    def __init__(self, input_dim, num_labels=7, dropout_prob=0.1, hidden_dim=None, use_crf=True):
        hidden_size = hidden_dim if hidden_dim is not None else input_dim
        super().__init__(
            input_dim=input_dim,
            num_labels=num_labels,
            hidden_size=hidden_size,
            num_lstm_layers=1,
            dropout=dropout_prob,
            use_crf=use_crf,
            enable_bilstm=False
        )
