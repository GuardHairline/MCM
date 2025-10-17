from datasets.mabsa_dataset import MABSADataset
from datasets.mnre_dataset import MNREDataset
from datasets.masc_dataset import MASCDataset
from datasets.mate_dataset import MATEDataset
from datasets.mner_dataset import MNERDataset
from datasets.deqa_dataset import MASCDatasetDEQA, MATEDatasetDEQA
from datasets.deqa_dataset_extended import MABSADatasetDEQA, MNERDatasetDEQA

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
        text_model_name = args.get("text_model_name")
        use_deqa = args.get("deqa", False)
        description_file = args.get("description_file", None)
    else:
        text_file = getattr(args, f"{split}_text_file")
        image_dir = args.image_dir
        text_model_name = args.text_model_name
        use_deqa = getattr(args, 'deqa', False)
        description_file = getattr(args, 'description_file', None)
    
    # 根据模型类型确定最大序列长度
    if "clip" in text_model_name.lower():
        max_seq_length = 77  # CLIP模型的最大序列长度
    else:
        max_seq_length = 128  # 其他模型的默认序列长度

    if task == "mabsa":
        if use_deqa:
            return MABSADatasetDEQA(
                text_file=text_file,
                image_dir=image_dir,
                tokenizer_name=text_model_name,
                max_seq_length=max_seq_length,
                description_file=description_file
            )
        else:
            return MABSADataset(
                text_file=text_file,
                image_dir=image_dir,
                tokenizer_name=text_model_name,
                max_seq_length=max_seq_length
            )
    elif task == "mner":
        if use_deqa:
            return MNERDatasetDEQA(
                text_file=text_file,
                image_dir=image_dir,
                tokenizer_name=text_model_name,
                max_seq_length=max_seq_length,
                description_file=description_file
            )
        else:
            return MNERDataset(
                text_file=text_file,
                image_dir=image_dir,
                tokenizer_name=text_model_name,
                max_seq_length=max_seq_length
            )
    elif task == "mnre":
        return MNREDataset(
            text_file=text_file,
            image_dir=image_dir,
            tokenizer_name=text_model_name,
            max_seq_length=max_seq_length
        )
    elif task == "mate":
        if use_deqa:
            return MATEDatasetDEQA(
                text_file=text_file,
                image_dir=image_dir,
                tokenizer_name=text_model_name,
                max_seq_length=max_seq_length,
                description_file=description_file
            )
        else:
            return MATEDataset(
                text_file=text_file,
                image_dir=image_dir,
                tokenizer_name=text_model_name,
                max_seq_length=max_seq_length
            )
    elif task == "masc":
        if use_deqa:
            return MASCDatasetDEQA(
                text_file=text_file,
                image_dir=image_dir,
                tokenizer_name=text_model_name,
                max_seq_length=max_seq_length,
                description_file=description_file
            )
        else:
            return MASCDataset(
                text_file=text_file,
                image_dir=image_dir,
                tokenizer_name=text_model_name,
                max_seq_length=max_seq_length
            )
    else:
        raise ValueError(f"Unsupported task: {task}")

