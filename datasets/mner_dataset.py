# datasets/mner_dataset.py
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms

class MNERDataset(Dataset):
    """
    MNER: Multimodal NER, 第三行为实体类型(-1..2), 做序列标注(B-type/I-type/O).
    若每条样本只有一个实体, 我们就对这个实体做B-type/I-type, 其余token为O.
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

        self.samples = self._read_data()

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 定义 type_map
        self.type_map = {-1: 0, 0:1, 1:2, 2:3}  # 4类

    def _read_data(self):
        lines = []
        with open(self.text_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        assert len(lines) % 4 == 0, "MNER 数据格式有误, 每4行构成一条样本"

        samples = []
        for i in range(0, len(lines), 4):
            text_with_T = lines[i]
            entity_str = lines[i+1]
            entity_type_str = lines[i+2]  # -1 0 1 2
            image_name = lines[i+3]
            if not image_name.endswith(".jpg"):
                image_name += ".jpg"
            image_path = os.path.join(self.image_dir, image_name)

            entity_type = int(entity_type_str)  # -1..2
            samples.append((text_with_T, entity_str, entity_type, image_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text_with_T, entity_str, entity_type, image_path = self.samples[idx]

        start_pos = text_with_T.index("$T$")
        end_pos = start_pos + len(entity_str) - 1
        replaced_text = text_with_T.replace("$T$", entity_str)

        # B-type / I-type / O=0
        t = self.type_map[entity_type]  # => t in [0..3]
        # B-type => 1 + 2*t
        # I-type => 2 + 2*t
        # O => 0
        char_label = [0]*len(replaced_text)
        if 0 <= start_pos < len(replaced_text):
            b_label = 1 + 2*t
            i_label = 2 + 2*t
            char_label[start_pos] = b_label
            for c in range(start_pos+1, end_pos+1):
                char_label[c] = i_label

        # tokenizer + offset
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
                label_ids.append(-100)
            else:
                sub_chars = char_label[start_char:end_char]
                if len(sub_chars) == 0:
                    label_ids.append(-100)
                else:
                    label_ids.append(sub_chars[0])

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
        with Image.open(path).convert("RGB") as img:
            return self.image_transform(img)
