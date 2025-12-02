# datasets/mate_dataset.py
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image
import os
import torchvision.transforms as transforms


class MATEDataset(Dataset):
    """
    用于 MATE (Multimodal Aspect Term Extraction) 任务的序列标注数据集。
    只关注方面词的抽取，不处理情感极性。
    数据格式假设与 MASC 相同，每个样本4行：
      1) 原文，带 $T$ 占位符
      2) aspect_term
      3) dummy_label(如 -1, 0, 1)，这里忽略
      4) image_name

    标签数量：O=0, B=1, I=2
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

    def _read_data(self):
        lines = []
        with open(self.text_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        assert len(lines) % 4 == 0, "MATE 数据格式有误，每条样本应占4行。"

        # 聚合字典: Key = Key = (clean_text, image_path), Value = list of Aspects
        grouped_data = {}

        for i in range(0, len(lines), 4):
            text_with_T = lines[i]
            aspect_term = lines[i + 1]
            _ = lines[i + 2]  # sentiment, 本任务用不到
            image_name = lines[i + 3]

            if not image_name.endswith(".jpg"):
                image_name += ".jpg"
            image_path = os.path.join(self.image_dir, image_name)

            # 清洗文本模板
            clean_template = " ".join(text_with_T.split())

            # 定位 $T$ 的位置
            parts = clean_template.split("$T$")
            if len(parts) < 2:
                # 如果数据行里居然没有 $T$，打印警告并跳过
                print(f"[Warning] No $T$ found in line: {text_with_T}")
                continue
                
            prefix = parts[0]

            start_idx = len(prefix)
            end_idx = start_idx + len(aspect_term) - 1

            # 还原标准化文本
            raw_text_clean = clean_template.replace("$T$", aspect_term)
            
            # 聚合 Key
            key = (raw_text_clean, image_path)

            # 2. 初始化聚合记录
            if key not in grouped_data:
                # 第一次遇到该图片，尝试还原完整文本
                # 注意：这里假设第一个遇到的实体的文本就是基准文本
                # 将 $T$ 替换为实体词
                grouped_data[key] = []
                
            grouped_data[key].append({
                "aspect": aspect_term,
                "start": start_idx,
                "end": end_idx
            })
        samples = []
        for (text, img_path), aspects in grouped_data.items():
            samples.append((text, img_path, aspects))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, image_path, aspects = self.samples[idx]

        # 初始化字符级标签（全0）
        char_label = [0] * len(text)

        for asp in aspects:
            start_pos = asp['start']
            end_pos = asp['end']
            
            # MATE: B=1, I=2
            if start_pos < len(text) and end_pos < len(text):
                char_label[start_pos] = 1 # B
                for c in range(start_pos + 1, end_pos + 1):
                    char_label[c] = 2 # I

        # Tokenizer
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
                label_ids.append(-100)
            else:
                sub_chars = char_label[start_char:end_char]
                token_label = 0
                # 只要由非0标签，优先取非0
                for l in sub_chars:
                    if l != 0:
                        # 简单逻辑：如果包含B(1)，则标为1；否则如果有I(2)，标为2
                        if l == 1: 
                            token_label = 1
                            break
                        token_label = 2
                label_ids.append(token_label)

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
            # print(f"Error loading image {path}: {e}")
            return torch.zeros(3, 224, 224)