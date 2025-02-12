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
    """

    def __init__(self,
                 text_file: str,
                 image_dir: str,
                 tokenizer_name: str = "microsoft/deberta-v3-base",
                 max_seq_length: int = 128):
        super().__init__()
        self.text_file = text_file
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length

        self.samples = []
        self._read_data()

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

        for i in range(0, len(lines), 4):
            text_with_T = lines[i]
            aspect_term = lines[i + 1]
            _ = lines[i + 2]  # sentiment, 本任务用不到
            image_name = lines[i + 3]

            if not image_name.endswith(".jpg"):
                image_name += ".jpg"
            image_path = os.path.join(self.image_dir, image_name)
            self.samples.append((text_with_T, aspect_term, image_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text_with_T, aspect_term, image_path = self.samples[idx]

        # 1) 替换 $T$ => new_text


        if "$T$" in text_with_T:
            replaced_text = text_with_T.replace("$T$", aspect_term)
            start_pos = replaced_text.index(aspect_term)
            end_pos = start_pos + len(aspect_term) - 1
        else:
            replaced_text = text_with_T  # 若无 $T$, 也可直接拼
            start_pos = -1
            end_pos = -1


        # 2) 构造 char_label (B=1, I=2, O=0)
        char_label = [0]*len(replaced_text)
        if 0 <= start_pos < len(replaced_text):
            char_label[start_pos] = 1  # B
            for c in range(start_pos+1, end_pos+1):
                char_label[c] = 2       # I

        # 3) tokenizer + offset对齐 => subword级别标签
        encoded = self.tokenizer(
            replaced_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True
        )
        offsets = encoded["offset_mapping"]

        label_ids = []
        for (start_char, end_char) in offsets:
            if start_char == end_char:
                # 特殊token [CLS], [SEP], or padding
                label_ids.append(-100)
            else:
                sub_labels  = char_label[start_char:end_char]
                valid_labels = [l for l in sub_labels if l != 0]

                if not valid_labels:
                    label_id = 0  # 无有效标签，标记为 O
                else:
                    # 策略1: 取出现最多的标签
                    # label_id = Counter(valid_labels).most_common(1)[0][0]

                    # 策略2: B 标签优先（若存在 B 则覆盖 I）
                    label_id = 1 if 1 in valid_labels else (
                        2 if 2 in valid_labels else 0
                    )
                label_ids.append(label_id)

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

    def _load_image(self, image_path):
        with Image.open(image_path).convert("RGB") as img:
            img_tensor = self.image_transform(img)
        return img_tensor