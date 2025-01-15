# \models\task_heads\get_head.py
from models.task_heads.mabsa_head import MABSAHead
from models.task_heads.mner_head import MNERHead
from models.task_heads.re_head import REHead
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
            num_labels=args.num_labels
        )
    elif task == "mner":
        return MNERHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=args.num_labels
        )
    elif task == "re":
        return REHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=args.num_labels
        )
    elif task == "mate":
        return MATEHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=args.num_labels
        )
    elif task == "masc":
        return MASCHead(
            input_dim=base_model.fusion_output_dim,
            num_labels=args.num_labels
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
