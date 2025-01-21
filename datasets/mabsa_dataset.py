# datasets/mabsa_dataset.py
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms

class MABSADataset(Dataset):
    """
    MABSA: End-to-End.
    假设1个样本1个aspect(但可扩展).
    7类序列标签: O=0, B-NEG=1,I-NEG=2, B-NEU=3,I-NEU=4, B-POS=5,I-POS=6
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

    def _read_data(self):
        lines = []
        with open(self.text_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        assert len(lines) % 4 == 0, "MABSA格式有误"

        samples = []
        for i in range(0, len(lines), 4):
            text_with_T = lines[i]
            aspect_term = lines[i+1]
            sentiment_str = lines[i+2]
            image_name = lines[i+3]
            sentiment = int(sentiment_str)  # -1->NEG,0->NEU,1->POS
            if not image_name.endswith(".jpg"):
                image_name += ".jpg"
            image_path = os.path.join(self.image_dir, image_name)

            samples.append((text_with_T, aspect_term, sentiment, image_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text_with_T, aspect_term, sentiment, image_path = self.samples[idx]

        replaced_text = text_with_T.replace("$T$", aspect_term)
        start_pos = replaced_text.find(aspect_term)
        end_pos = start_pos + len(aspect_term) - 1

        # sentiment => -1 => NEG= (B=1,I=2), 0 => NEU=(3,4), 1 => POS=(5,6)
        # O => 0
        if sentiment == -1:
            b_val, i_val = 1, 2
        elif sentiment == 0:
            b_val, i_val = 3, 4
        else:  # ==1
            b_val, i_val = 5, 6

        char_label = [0]*len(replaced_text)
        if 0 <= start_pos < len(replaced_text):
            char_label[start_pos] = b_val
            for c in range(start_pos+1, end_pos+1):
                char_label[c] = i_val

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
