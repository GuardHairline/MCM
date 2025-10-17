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

        self.samples = []
        self._read_data()
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _read_data(self):
        """
        读取 text_file 文件，每4行为一个样本：原文、方面词、情感和图像路径。
        """
        lines = []
        with open(self.text_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        assert len(lines) % 4 == 0, "MABSA 数据格式有误，每条样本应占4行。"

        for i in range(0, len(lines), 4):
            text_with_T = lines[i]
            aspect_term = lines[i + 1]
            sentiment_str = lines[i + 2]
            image_name = lines[i + 3]

            if not image_name.endswith(".jpg"):
                image_name += ".jpg"
            image_path = os.path.join(self.image_dir, image_name)

            sentiment = int(sentiment_str)  # -1,0,1
            self.samples.append((text_with_T, aspect_term, sentiment, image_path))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text_with_T, aspect_term, sentiment, image_path = self.samples[idx]

        # 替换 $T$ => aspect_term
        if "$T$" in text_with_T:
            T_position = text_with_T.index("$T$")
            replaced_text = text_with_T.replace("$T$", aspect_term)
            start_pos = T_position
            end_pos = start_pos + len(aspect_term) - 1
        else:
            replaced_text = text_with_T  # 若无 $T$, 也可直接拼
            start_pos = -1
            end_pos = -1

        # 对每个字/词进行标注（B-情感、I-情感、O）
        char_label = [0] * len(replaced_text)  # 0 for O (non-aspect)

        # 根据 sentiment 生成 B 和 I 标签
        if aspect_term in replaced_text:
            sentiment_label = self._get_sentiment_label(sentiment)
            char_label[start_pos] = sentiment_label[0]  # B-情感
            for i in range(start_pos + 1, end_pos + 1):
                char_label[i] = sentiment_label[1]  # I-情感
        assert any(l != 0 for l in char_label), "Entity not labeled!"

        # Tokenize the text
        tokenized_input = self.tokenizer(replaced_text,
                                         max_length=self.max_seq_length,
                                         padding='max_length',
                                         truncation=True,
                                         return_offsets_mapping=True)
        offsets = tokenized_input["offset_mapping"]

        # Align labels with tokens
        label_ids = self._align_labels_with_tokens(offsets, char_label)
        assert any(label in {1, 2, 3, 4, 5, 6} for label in label_ids), f"No valid entity labels (1-6) found. Check text: '{replaced_text}', entity: '{aspect_term}', type: {char_label}"

        image_tensor = self._load_image(image_path)

        return {
            "input_ids": torch.tensor(tokenized_input["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized_input["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "image_tensor": image_tensor
        }

    def _get_sentiment_label(self, sentiment):
        """
        根据情感值返回对应的 B 和 I 标签。
        """
        labels = get_label_manager().get_sentiment_labels("mabsa", sentiment)
        if labels is None:
            raise ValueError(f"Invalid sentiment value: {sentiment} for MABSA task")
        return labels

    def _align_labels_with_tokens(self, offsets, char_label):
        """
        对齐字符级标签与token级标签。
        """
        label_ids = []
        for start_char, end_char in offsets:
            if start_char == end_char:
                label_ids.append(-100)  # Special tokens (CLS, SEP)
            else:
                sub_labels = char_label[start_char:end_char]
                valid_labels = [l for l in sub_labels if l != 0]
                label_ids.append(valid_labels[0] if valid_labels else 0)  # Use O if no valid label

        return label_ids

    def _load_image(self, image_path):
        with Image.open(image_path).convert("RGB") as img:
            return self.image_transform(img)
