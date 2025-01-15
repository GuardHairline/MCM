# datasets/masc_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer


class MASCDataset(Dataset):
    def __init__(self, text_file, image_dir, tokenizer_name, max_length=128, transform=None):
        """
        :param text_file: 文本文件路径
        :param image_dir: 存放图片的目录
        :param tokenizer_name: 用于文本 tokenization 的模型，比如 'bert-base-chinese'
        :param max_length: 文本序列最大长度
        :param transform: 图像预处理 transform
        """
        self.text_file = text_file
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # 读取数据
        self.data = self._parse_text_data()

    def _parse_text_data(self):
        data_list = []
        with open(self.text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i in range(0, len(lines), 4):
            text = lines[i].strip()
            entity = lines[i + 1].strip()
            sentiment = int(lines[i + 2].strip())
            image_name = lines[i + 3].strip()

            if not image_name.endswith(".jpg"):
                image_name += ".jpg"
            image_path = os.path.join(self.image_dir, image_name)

            # 记录
            data_list.append((text, entity, sentiment, image_path))
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, entity, sentiment, image_path = self.data[idx]

        # 文本处理
        # 这里假设我们把 text 和 entity 连接起来（也可以视需求分开编码）
        # 你也可以只对 text 做编码，把 entity 作为额外 token
        input_text = f"{text} [SEP] {entity}"
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 图像处理
        # 需要先判断文件是否存在，以及能否被PIL打开
        try:
            with Image.open(image_path).convert('RGB') as img:
                img_tensor = self.transform(img)
        except:
            # 若图片损坏，可返回一个空白或随机tensor
            img_tensor = torch.zeros((3, 224, 224))

        # sentiment 标签：转换为模型可接受的Tensor，并映射到 [0, 2]
        label = torch.tensor(sentiment + 1, dtype=torch.long)  # 将 -1, 0, 1 映射到 0, 1, 2


        # 返回一个字典或元组
        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", None).squeeze(0) if encoding.get("token_type_ids", None) is not None else None,  # [128] 或 None
            "image_tensor": img_tensor,  # [3, 224, 224]
            "label": label,
            "entity": entity,  # 有时在推理分析时可查看
            "text": text
        }
