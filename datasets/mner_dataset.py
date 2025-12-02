# datasets/mner_dataset.py
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms

class MNERDataset(Dataset):
    """
    MNER: Multimodal NER, 第三行为实体类型(-1..2), 做序列标注(B-type/I-type/O).
    若每条样本只有一个实体, 我们就对这个实体做B-type/I-type, 其余token为O.
    标签：O=0, B-PER=1, I-PER=2, B-ORG=3, I-ORG=4, B-LOC=5, I-LOC=6, B-MISC=7, I-MISC=8
    """

    def __init__(self,
                 text_file: str,
                 image_dir: str,
                 tokenizer_name: str = "microsoft/deberta-v3-base",
                 max_seq_length: int = 128):
        super().__init__()
        if tokenizer_name == "microsoft/deberta-v3-base":
            model_path = "downloaded_model/deberta-v3-base"
        elif "clip" in tokenizer_name.lower():
            model_path = tokenizer_name  # 直接使用CLIP模型名称
        else:
            model_path = tokenizer_name  # 默认使用传入的模型名称
        self.text_file = text_file
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length

        self.samples = self._read_data()

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 定义 type_map
        #"PER": "-1", "ORG": "0", "LOC": "1", "OTHER": "2", "MISC": "3"}
        self.type_map = {-1: 0, 0:1, 1:2, 2:3, 3:3}  # 4类

    def _read_data(self):
        lines = []
        with open(self.text_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        assert len(lines) % 4 == 0, "MNER 数据格式有误, 每4行构成一条样本"
        # 字典用于聚合：Key = (clean_text, image_path), Value = list of entities
        grouped_data = {}

        for i in range(0, len(lines), 4):
            text_with_T = lines[i]
            entity_str = lines[i+1]
            entity_type_str = lines[i+2]  # -1 0 1 2
            image_name = lines[i+3]
            
            if not image_name.endswith(".jpg"):
                image_name += ".jpg"
            image_path = os.path.join(self.image_dir, image_name)


            # 先对含 $T$ 的模板文本进行标准化清洗
            # .split() 会自动去除 \t, \n 和多余空格，" ".join 将其重建为标准空格分隔的字符串
            clean_template = " ".join(text_with_T.split())
            
            parts = clean_template.split("$T$")
            
            if len(parts) < 2:
                # 如果数据行里居然没有 $T$，打印警告并跳过
                print(f"[Warning] No $T$ found in line: {text_with_T}")
                continue
                
            prefix = parts[0]

            start_idx = len(prefix)
            end_idx = start_idx + len(entity_str) - 1

            # 还原标准化文本
            raw_text_clean = clean_template.replace("$T$", entity_str)
            
            key = (raw_text_clean, image_path)
            
            if key not in grouped_data:
                grouped_data[key] = []
            
            grouped_data[key].append({
                "entity": entity_str,
                "type": int(entity_type_str),# -1..2
                "start": start_idx,
                "end": end_idx
            })
        samples = []
        for (text, img_path), entities in grouped_data.items():
            samples.append((text, img_path, entities))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, image_path, entities = self.samples[idx]

        # 初始化字符级标签（全0）
        char_label = [0] * len(text)
        
        # 遍历该句子下的【所有】实体进行标注
        for ent in entities:
            start_pos = ent['start']
            end_pos = ent['end']
            entity_type = ent['type']
            
            # 映射类型
            t = self.type_map[entity_type]
            b_label = 1 + 2*t
            i_label = 2 + 2*t
            
            # 边界检查
            if start_pos < len(text) and end_pos < len(text):
                # 标记 B
                # 简单的冲突处理：如果该位置已经被标记过（非0），这里选择覆盖（或者跳过）
                # 通常 Twitter 数据集无嵌套，直接覆盖即可
                char_label[start_pos] = b_label
                
                # 标记 I
                for c in range(start_pos + 1, end_pos + 1):
                    char_label[c] = i_label

        # tokenizer + offset
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True
        )
        offsets = encoded["offset_mapping"]

        label_ids = []
        for (start_char, end_char) in offsets:
            if start_char == end_char:
                # 特殊 token ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            else:
                # 只要该 token 范围内有非 0 标签，就取非 0 的（简单策略）
                sub_chars = char_label[start_char:end_char]

                # 统计 sub_chars 里的标签
                # 优先取 B-tag，其次 I-tag
                token_label = 0
                for l in sub_chars:
                    if l != 0:
                        token_label = l
                        break # 只要碰到实体标记就认为是实体
                
                label_ids.append(token_label)
        assert any(label in {1, 2, 3, 4, 5, 6, 7, 8} for label in label_ids), f"No valid entity labels (1-8) found. Check text: '{replaced_text}', entity: '{entity_str}', type: {entity_type}"
        encoded.pop("offset_mapping")

        image_tensor = self._load_image(image_path)

        out_item = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "image_tensor": image_tensor
        }
        if "token_type_ids" in encoded:
            out_item["token_type_ids"] = torch.tensor(encoded["token_type_ids"], dtype=torch.long)

        return out_item

    def _load_image(self, path):
        try:
            with Image.open(path).convert("RGB") as img:
                return self.image_transform(img)
        except (UnidentifiedImageError, IOError) as e:
            print(f"Error loading image {path}: {e}")
            # 返回一个默认的全零图像（假设尺寸为 224x224，3通道）
            return torch.zeros(3, 224, 224)
