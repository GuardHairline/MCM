# \models\task_heads\get_head.py
from models.task_heads.mabsa_head import MABSAHead
from models.task_heads.mner_head import MNERHead
from models.task_heads.mnre_head import MNREHead
from models.task_heads.masc_head import MASCHead
from models.task_heads.mate_head import MATEHead
def get_head(task, base_model, args):
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
            num_labels=args.num_labels,
            dropout_prob=args.dropout_prob
        )
    elif task == "mner":
        return MNERHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=args.num_labels,
            dropout_prob=args.dropout_prob
        )
    elif task == "mnre":
        return MNREHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=args.num_labels
        )
    elif task == "mate":
        return MATEHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=args.num_labels,
            dropout_prob=args.dropout_prob,  # 传入 dropout 参数
            hidden_dim=args.hidden_dim  # 传入 hidden_dim 参数
        )
    elif task == "masc":
        return MASCHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=args.num_labels
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
