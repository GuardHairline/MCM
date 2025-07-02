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
def get_head(task, base_model, args, label_emb=None):
    if isinstance(args, dict):
        num_labels = args.get("num_labels")
        dropout_prob = args.get("dropout_prob")
        hidden_dim = args.get("hidden_dim")
    else:
        num_labels = args.num_labels
        dropout_prob = args.dropout_prob
        hidden_dim = args.hidden_dim
    """
    根据任务初始化相应的任务头
    :param task: 任务名称
    :param base_model: 基础模型对象
    :param args: 参数对象
    :return: 任务头对象
    """
    if task == "mabsa":
        return MABSAHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=num_labels,
            dropout_prob=dropout_prob,
            hidden_dim=hidden_dim
        )
    elif task == "mner":
        return MNERHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=num_labels,
            dropout_prob=dropout_prob,
            hidden_dim=hidden_dim
        )
    # elif task == "mnre":
    #     return MNREHead(
    #         input_dim=base_model.fusion_output_dim,
    #         num_labels=num_labels,
    #         dropout_prob=dropout_prob,  # 传入 dropout 参数
    #         hidden_dim=hidden_dim  # 传入 hidden_dim 参数
    #     )
    elif task == "mate":
        return MATEHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=num_labels,
            dropout_prob=dropout_prob,  # 传入 dropout 参数
            hidden_dim=hidden_dim  # 传入 hidden_dim 参数
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
