# datasets/mnre_dataset.py
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms

class MNREDataset(Dataset):
    """
    MNRE: 多模态关系抽取.
    假设8行=1个样本:
      [0] text_with_T (头实体那份)
      [1] head_entity
      [2] relation_label
      [3] image_name
      [4] text_with_T (尾实体那份)
      [5] tail_entity
      [6] relation_label
      [7] image_name
    => 构造: "text_with_two_T".replace("$T$", "<head> head_entity </head>", 第一次出现)
       再替换第二次 "$T$" => "<tail> tail_entity </tail>"
    => 关系分类 => (batch_size, num_relations)
    """

    def __init__(self,
                 text_file: str,
                 image_dir: str,
                 tokenizer_name: str = "microsoft/deberta-v3-base",
                 max_seq_length: int = 128,
                 num_relations: int = 23):
        super().__init__()
        if tokenizer_name == "microsoft/deberta-v3-base":
            model_path = "downloaded_model/deberta-v3-base"
        self.text_file = text_file
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length
        self.num_relations = num_relations

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
        assert len(lines) % 8 == 0, "MNRE 数据应8行为一个样本(头+尾)"

        samples = []
        for i in range(0, len(lines), 8):
            text_head = lines[i]      # 带 $T$(头)
            head_entity = lines[i+1]
            relation_label_str = lines[i+2]
            image_name1 = lines[i+3]

            text_tail = lines[i+4]    # 带 $T$(尾)
            tail_entity = lines[i+5]
            relation_label_str2 = lines[i+6]
            image_name2 = lines[i+7]

            relation_label = int(relation_label_str)

            label_id = relation_label + 1  # 这样 -1->0, 22->23

            if not image_name1.endswith(".jpg"):
                image_name1 += ".jpg"
            image_path = os.path.join(self.image_dir, image_name1)

            # 第一次替换:
            merged_text = text_head.replace("$T$", f"<head> {head_entity} </head>")
            # 第二次替换:
            merged_text = merged_text.replace(tail_entity, f"<tail> {tail_entity} </tail>")

            samples.append((merged_text, label_id, image_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        merged_text, label_id, image_path = self.samples[idx]

        # 句级分类 => 直接 tokenizer => input_ids
        encoded = self.tokenizer(
            merged_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True
        )
        image_tensor = self._load_image(image_path)

        out_item = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label_id, dtype=torch.long),
            "image_tensor": image_tensor
        }
        if "token_type_ids" in encoded:
            out_item["token_type_ids"] = torch.tensor(encoded["token_type_ids"], dtype=torch.long)

        return out_item

    def _load_image(self, path):
        with Image.open(path).convert("RGB") as img:
            return self.image_transform(img)
