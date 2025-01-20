# datasets/masc_dataset.py
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image
import os
import torchvision.transforms as transforms


class MASCDataset(Dataset):
    """
    用于 MASC (Multimodal Aspect Sentiment Classification) 任务的序列标注数据集。
    数据格式假设为每个样本四行：
      1) 原文，带 $T$ 占位符
      2) aspect_term (替换 $T$ 的真实字符串)
      3) sentiment (可能是 -1, 0, 1)
      4) image_name (图像文件名)
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

        # 定义图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _read_data(self):
        """
        读取 text_file 文件，每 4 行构成一个样本。
        """
        lines = []
        with open(self.text_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        # 以4行为一组
        assert len(lines) % 4 == 0, "MASC 数据格式有误，每条样本应占4行。"

        for i in range(0, len(lines), 4):
            text_with_T = lines[i]
            aspect_term = lines[i+1]
            sentiment_str = lines[i+2]
            image_name = lines[i+3]

            if not image_name.endswith(".jpg"):
                image_name += ".jpg"
            image_path = os.path.join(self.image_dir, image_name)

            sentiment = int(sentiment_str)  # -1,0,1

            # 记录
            self.samples.append((text_with_T, aspect_term, sentiment, image_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text_with_T, aspect_term, sentiment, image_path = self.samples[idx]

        # 1) 替换 $T$ => 在这里插入可视Marker，如 <asp> aspect_term </asp>
        #    让模型能明显区分出aspect位置
        if "$T$" in text_with_T:
            replaced_text = text_with_T.replace("$T$", f"<asp> {aspect_term} </asp>")
        else:
            # 如果文本本来就没有$T$, 也可以直接拼在后面
            replaced_text = text_with_T + f" <asp> {aspect_term} </asp>"

        # 2) 句级情感分类 => 不做序列标注
        #    只需要将 replaced_text 编码为 input_ids, attention_mask
        tokenized_input = self.tokenizer(
            replaced_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True
        )

        # 3) 读取并处理图像
        image_tensor = self._load_image(image_path)

        # 4) 返回句级情感标签(三分类): sentiment => { -1 -> 0, 0 -> 1, 1 -> 2 } (可选做一个偏移映射)
        #    或者直接保留 -1,0,1, 训练时让 head 输出维度=3 并自己写个 label->[0,1,2] 的映射也行
        label_map = {-1: 0, 0: 1, 1: 2}
        label_id = label_map[sentiment]

        out_item = {
            "input_ids": torch.tensor(tokenized_input["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized_input["attention_mask"], dtype=torch.long),
            "image_tensor": image_tensor,
            "labels": torch.tensor(label_id, dtype=torch.long)  # 三分类
        }

        if "token_type_ids" in tokenized_input:
            out_item["token_type_ids"] = torch.tensor(tokenized_input["token_type_ids"], dtype=torch.long)
        # 5) cross_labels 只做预留, 这里可以不返回或返回空
        out_item["cross_labels"] = torch.zeros_like(out_item["input_ids"])

        return out_item

    def _load_image(self, image_path):
        # 简易的图像读取 + transform
        with Image.open(image_path).convert("RGB") as img:
            img_tensor = self.image_transform(img)
        return img_tensor