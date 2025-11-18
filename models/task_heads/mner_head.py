from models.task_heads.mner_head_bilstm import MNERHeadBiLSTM


class MNERHead(MNERHeadBiLSTM):
    """
    兼容旧接口的MNER任务头。
    使用BiLSTM版本的实现，但禁用BiLSTM部分，仅保留投影+CRF。
    """

    def __init__(self, input_dim, num_labels, dropout_prob=0.1, hidden_dim=None, use_crf=True):
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
