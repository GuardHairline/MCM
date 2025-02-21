from datasets.mabsa_dataset import MABSADataset
from datasets.mnre_dataset import MNREDataset
from datasets.masc_dataset import MASCDataset
from datasets.mate_dataset import MATEDataset
from datasets.mner_dataset import MNERDataset

def get_dataset(task, split, args):
    """
    根据任务和数据分割加载相应的数据集
    :param task: 任务名称
    :param split: 数据分割名称（train, dev, test）
    :param args: 参数对象
    :return: 数据集对象
    """
    if isinstance(args, dict):
        text_file = args.get(f"{split}_text_file")
        image_dir = args.get("image_dir")
    else:
        text_file = getattr(args, f"{split}_text_file")
        image_dir = args.image_dir

    if task == "mabsa":
        return MABSADataset(
            text_file=text_file,
            image_dir=image_dir,
            tokenizer_name=args.text_model_name
        )
    elif task == "mner":
        return MNERDataset(
            text_file=text_file,
            image_dir=image_dir,
            tokenizer_name=args.text_model_name
        )
    elif task == "mnre":
        return MNREDataset(
            text_file=text_file,
            image_dir=image_dir,
            tokenizer_name=args.text_model_name
        )
    elif task == "mate":
        return MATEDataset(
            text_file=text_file,
            image_dir=image_dir,
            tokenizer_name=args.text_model_name
        )
    elif task == "masc":
        return MASCDataset(
            text_file=text_file,
            image_dir=image_dir,
            tokenizer_name=args.text_model_name
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

