# datasets/mabsa_dataset.py
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms
from continual.label_config import get_label_manager

class MABSADataset(Dataset):
    """
    MABSA: Multimodal Aspect-Based Sentiment Analysis.
    标签体系 (7类): 
    O=0
    Negative(-1) -> B=1, I=2
    Neutral(0)   -> B=3, I=4
    Positive(1)  -> B=5, I=6
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
        # 情感映射表: sentiment_val -> base_index
        # -1(Neg) -> base 0 -> B=1, I=2
        #  0(Neu) -> base 1 -> B=3, I=4
        #  1(Pos) -> base 2 -> B=5, I=6
        self.sentiment_map = {-1: 0, 0: 1, 1: 2}    
    def _read_data(self):
        """
        读取 text_file 文件，每4行为一个样本：原文、方面词、情感和图像路径。
        """
        lines = []
        with open(self.text_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        assert len(lines) % 4 == 0, "MABSA 数据格式有误，每条样本应占4行。"

        grouped_data = {}
        
        for i in range(0, len(lines), 4):
            text_with_T = lines[i]
            aspect_term = lines[i+1]
            sentiment_str = lines[i+2]
            image_name = lines[i+3]
            
            if not image_name.endswith(".jpg"):
                image_name += ".jpg"
            image_path = os.path.join(self.image_dir, image_name)

            clean_template = " ".join(text_with_T.split())
            parts = clean_template.split("$T$")
            if len(parts) < 2:
                print(f"[Warning] No $T$ found in line: {text_with_T}")
                continue
            
            prefix = parts[0]
            start_idx = len(prefix)
            end_idx = start_idx + len(aspect_term) - 1

            raw_text_clean = clean_template.replace("$T$", aspect_term)
            
            key = (raw_text_clean, image_path)
            if key not in grouped_data:
                grouped_data[key] = []
            
            grouped_data[key].append({
                "aspect": aspect_term,
                "sentiment": int(sentiment_str), # -1, 0, 1
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

        char_label = [0] * len(text)
        
        for asp in aspects:
            start_pos = asp['start']
            end_pos = asp['end']
            sentiment = asp['sentiment']
            
            # 计算标签ID
            if sentiment in self.sentiment_map:
                base = self.sentiment_map[sentiment]
                b_label = 1 + 2 * base
                i_label = 2 + 2 * base
                
                if start_pos < len(text) and end_pos < len(text):
                    char_label[start_pos] = b_label
                    for c in range(start_pos + 1, end_pos + 1):
                        char_label[c] = i_label

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
                for l in sub_chars:
                    if l != 0:
                        token_label = l
                        break 
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
            return torch.zeros(3, 224, 224)
