# \models\task_heads\get_head.py
from models.task_heads.mabsa_head import MABSAHead
from models.task_heads.mner_head import MNERHead
from models.task_heads.mnre_head import MNREHead
from models.task_heads.masc_head import MASCHead
from models.task_heads.mate_head import MATEHead
from models.task_heads.absa_head import ABSAHead
from models.task_heads.ner_head import NERHead
from models.task_heads.asc_head import ASCHead
from models.task_heads.ate_head import ATEHead

# BiLSTM增强版本 (推荐，性能更好)
from models.task_heads.mabsa_head_bilstm import MABSAHeadBiLSTM
from models.task_heads.mner_head_bilstm import MNERHeadBiLSTM
from models.task_heads.mate_head_bilstm import MATEHeadBiLSTM
def get_head(task, base_model, args, label_emb=None):
    """
    根据任务初始化相应的任务头
    
    :param task: 任务名称
    :param base_model: 基础模型对象
    :param args: 参数对象
    :param label_emb: 标签嵌入（可选）
    :return: 任务头对象
    
    说明：
    - 序列标注任务（mner, mate, mabsa）默认使用BiLSTM增强版本（性能更好）
    - 可以通过 args.use_bilstm=False 来禁用BiLSTM
    - BiLSTM参数：hidden_size=256, num_lstm_layers=2, dropout=0.3
    """
    if isinstance(args, dict):
        num_labels = args.get("num_labels")
        dropout_prob = args.get("dropout_prob", 0.3)
        hidden_dim = args.get("hidden_dim")
        use_crf = bool(args.get("use_crf", 1))  # 默认启用CRF
        use_bilstm = bool(args.get("use_bilstm", 1))  # 默认启用BiLSTM
        enable_bilstm_head = bool(args.get("enable_bilstm_head", 1))
        bilstm_hidden = args.get("bilstm_hidden_size", 256)
        bilstm_layers = args.get("bilstm_num_layers", 2)
    else:
        num_labels = args.num_labels
        dropout_prob = getattr(args, 'dropout_prob', 0.3)
        hidden_dim = args.hidden_dim
        use_crf = bool(getattr(args, 'use_crf', 1))  # 默认启用CRF
        use_bilstm = bool(getattr(args, 'use_bilstm', 1))  # 默认启用BiLSTM
        enable_bilstm_head = bool(getattr(args, 'enable_bilstm_head', 1))
        bilstm_hidden = getattr(args, 'bilstm_hidden_size', 256)
        bilstm_layers = getattr(args, 'bilstm_num_layers', 2)
    
    use_bilstm = use_bilstm and enable_bilstm_head
    
    if task == "mabsa":
        if use_bilstm:
            # BiLSTM增强版本（推荐）
            return MABSAHeadBiLSTM(
                input_dim=base_model.fusion_output_dim,
                num_labels=num_labels,
                hidden_size=bilstm_hidden,
                num_lstm_layers=bilstm_layers,
                dropout=dropout_prob,
                use_crf=use_crf
            )
        else:
            # 原始版本
            return MABSAHead(
                input_dim=base_model.fusion_output_dim,
                num_labels=num_labels,
                dropout_prob=dropout_prob,
                hidden_dim=hidden_dim,
                use_crf=use_crf
            )
    elif task == "mner":
        if use_bilstm:
            # BiLSTM增强版本（推荐）
            return MNERHeadBiLSTM(
                input_dim=base_model.fusion_output_dim,
                num_labels=num_labels,
                hidden_size=bilstm_hidden,
                num_lstm_layers=bilstm_layers,
                dropout=dropout_prob,
                use_crf=use_crf
            )
        else:
            # 原始版本
            return MNERHead(
                input_dim=base_model.fusion_output_dim,
                num_labels=num_labels,
                dropout_prob=dropout_prob,
                hidden_dim=hidden_dim,
                use_crf=use_crf
            )
    # elif task == "mnre":
    #     return MNREHead(
    #         input_dim=base_model.fusion_output_dim,
    #         num_labels=num_labels,
    #         dropout_prob=dropout_prob,  # 传入 dropout 参数
    #         hidden_dim=hidden_dim  # 传入 hidden_dim 参数
    #     )
    elif task == "mate":
        if use_bilstm:
            # BiLSTM增强版本（推荐）
            return MATEHeadBiLSTM(
                input_dim=base_model.fusion_output_dim,
                num_labels=num_labels,
                hidden_size=bilstm_hidden,
                num_lstm_layers=bilstm_layers,
                dropout=dropout_prob,
                use_crf=use_crf
            )
        else:
            # 原始版本
            return MATEHead(
                input_dim=base_model.fusion_output_dim,
                num_labels=num_labels,
                dropout_prob=dropout_prob,
                hidden_dim=hidden_dim,
                use_crf=use_crf
            )
    elif task == "masc":
        return MASCHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=num_labels,
            dropout_prob=dropout_prob,  # 传入 dropout 参数
            hidden_dim=hidden_dim  # 传入 hidden_dim 参数
        )
    elif task == "absa":
        return ABSAHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=num_labels,
            dropout_prob=dropout_prob,  # 传入 dropout 参数
            hidden_dim=hidden_dim  # 传入 hidden_dim 参数
        )
    elif task == "ate":
        return ATEHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=num_labels,
            dropout_prob=dropout_prob,  # 传入 dropout 参数
            hidden_dim=hidden_dim  # 传入 hidden_dim 参数
        )
    elif task == "asc":
        return ASCHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=num_labels,
            dropout_prob=dropout_prob,  # 传入 dropout 参数
            hidden_dim=hidden_dim  # 传入 hidden_dim 参数
        )
    elif task == "ner":
        return NERHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=num_labels,
            dropout_prob=dropout_prob,  # 传入 dropout 参数
            hidden_dim=hidden_dim  # 传入 hidden_dim 参数
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
