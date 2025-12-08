# datasets/deqa_dataset.py
"""
DEQA数据集 - 支持图像描述的数据集

扩展现有数据集以支持图像描述，用于DEQA方法
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image
import os
import torchvision.transforms as transforms
from typing import Optional

from datasets.description_manager import ImageDescriptionManager



class DEQADatasetMixin:
    """
    DEQA数据集Mixin - 为现有数据集添加描述支持
    
    在原有数据集基础上添加:
    - description_input_ids
    - description_attention_mask
    """
    
    def __init__(
        self,
        description_file: Optional[str] = None,
        description_tokenizer_name: Optional[str] = None,
        description_max_length: int = 256,
        **kwargs
    ):
        """
        Args:
            description_file: 描述文件路径
            description_tokenizer_name: 描述编码器名称（默认与文本编码器相同）
            description_max_length: 描述最大长度
        """
        # 加载描述管理器
        self.description_manager = None
        if description_file and os.path.exists(description_file):
            self.description_manager = ImageDescriptionManager(description_file)
        
        # 描述tokenizer
        self.description_max_length = description_max_length
        if description_tokenizer_name is None:
            description_tokenizer_name = kwargs.get('tokenizer_name', 'microsoft/deberta-v3-base')
        
        if description_tokenizer_name == "microsoft/deberta-v3-base":
            model_path = "downloaded_model/deberta-v3-base"
        else:
            model_path = description_tokenizer_name
        
        self.description_tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def get_description_encoding(self, image_name: str):
        """
        获取图像描述的编码
        
        Args:
            image_name: 图像文件名
            
        Returns:
            description_input_ids, description_attention_mask
        """
        # 获取描述文本
        if self.description_manager is not None:
            description = self.description_manager.get_description(image_name)
        else:
            description = "No description available."
        
        # 编码描述
        encoded = self.description_tokenizer(
            description,
            max_length=self.description_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)


class MASCDatasetDEQA(Dataset):
    """
    MASC数据集 + DEQA支持（图像描述）
    """
    
    def __init__(
        self,
        text_file: str,
        image_dir: str,
        tokenizer_name: str = "microsoft/deberta-v3-base",
        max_seq_length: int = 128,
        description_file: Optional[str] = None,
        description_max_length: int = 256
    ):
        super().__init__()
        
        # 文本tokenizer
        if tokenizer_name == "microsoft/deberta-v3-base":
            model_path = "downloaded_model/deberta-v3-base"
        else:
            model_path = tokenizer_name
        
        self.text_file = text_file
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length
        
        # 描述支持
        self.description_manager = None
        if description_file and os.path.exists(description_file):
            self.description_manager = ImageDescriptionManager(description_file)
        
        self.description_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.description_max_length = description_max_length
        
        # 读取数据
        self.samples = []
        self._read_data()
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _read_data(self):
        """读取数据"""
        lines = []
        with open(self.text_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        
        assert len(lines) % 4 == 0, "MASC 数据格式有误，每条样本应占4行。"
        
        for i in range(0, len(lines), 4):
            text_with_T = lines[i]
            aspect_term = lines[i+1]
            sentiment_str = lines[i+2]
            image_name = lines[i+3]
            
            if not image_name.endswith(".jpg"):
                image_name += ".jpg"
            image_path = os.path.join(self.image_dir, image_name)
            
            sentiment = int(sentiment_str)  # -1, 0, 1
            
            self.samples.append((text_with_T, aspect_term, sentiment, image_name, image_path))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text_with_T, aspect_term, sentiment, image_name, image_path = self.samples[idx]
        
        # 1. 替换 $T$
        if "$T$" in text_with_T:
            replaced_text = text_with_T.replace("$T$", f"<asp> {aspect_term} </asp>")
        else:
            replaced_text = text_with_T + f" <asp> {aspect_term} </asp>"
        
        # 2. 编码文本
        tokenized_input = self.tokenizer(
            replaced_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 3. 加载图像
        image_tensor = self._load_image(image_path)
        
        # 4. 编码描述
        description_input_ids, description_attention_mask = self._get_description_encoding(image_name)
        
        # 5. 标签转换 (MASC: -1,0,1 -> 0,1,2)
        label = sentiment + 1
        
        return {
            'input_ids': tokenized_input['input_ids'].squeeze(0),
            'attention_mask': tokenized_input['attention_mask'].squeeze(0),
            'token_type_ids': tokenized_input.get('token_type_ids', torch.zeros_like(tokenized_input['input_ids'])).squeeze(0),
            'image_tensor': image_tensor,
            'description_input_ids': description_input_ids,
            'description_attention_mask': description_attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """加载和预处理图像"""
        try:
            if not os.path.exists(image_path):
                # 创建空白图像
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            else:
                image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回空白图像
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            return self.image_transform(image)
    
    def _get_description_encoding(self, image_name: str):
        """获取图像描述的编码"""
        # 获取描述文本
        if self.description_manager is not None:
            description = self.description_manager.get_description(image_name)
        else:
            description = "No description available."
        
        # 编码描述
        encoded = self.description_tokenizer(
            description,
            max_length=self.description_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)


class MATEDatasetDEQA(Dataset):
    """
    MATE数据集 + DEQA支持（图像描述）
    """
    
    def __init__(
        self,
        text_file: str,
        image_dir: str,
        tokenizer_name: str = "microsoft/deberta-v3-base",
        max_seq_length: int = 128,
        description_file: Optional[str] = None,
        description_max_length: int = 256
    ):
        super().__init__()
        
        # 文本tokenizer
        if tokenizer_name == "microsoft/deberta-v3-base":
            model_path = "downloaded_model/deberta-v3-base"
        else:
            model_path = tokenizer_name
        
        self.text_file = text_file
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length
        
        # 描述支持
        self.description_manager = None
        if description_file and os.path.exists(description_file):
            self.description_manager = ImageDescriptionManager(description_file)
        
        self.description_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.description_max_length = description_max_length
        
        self.samples = self._read_data()

        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _read_data(self):
        """读取数据"""
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
        
        #  加载图像
        image_tensor = self._load_image(image_path)
        
        # 6. 编码描述
        image_name = os.path.basename(image_path)
        description_input_ids, description_attention_mask = self._get_description_encoding(image_name)
        
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            'token_type_ids': encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])).squeeze(0),
            'image_tensor': image_tensor,
            'description_input_ids': description_input_ids,
            'description_attention_mask': description_attention_mask,
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            # 容错处理：返回黑图
            return torch.zeros(3, 224, 224)
    
    def _get_description_encoding(self, image_name: str):
        """获取图像描述的编码"""
        if self.description_manager is not None:
            description = self.description_manager.get_description(image_name)
        else:
            description = "No description available."
        
        encoded = self.description_tokenizer(
            description,
            max_length=self.description_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

class MABSADatasetDEQA(Dataset):
    """
    MABSA数据集 + DEQA支持（图像描述）
    """
    
    def __init__(
        self,
        text_file: str,
        image_dir: str,
        tokenizer_name: str = "microsoft/deberta-v3-base",
        max_seq_length: int = 128,
        description_file: Optional[str] = None,
        description_max_length: int = 256
    ):
        super().__init__()
        
        # 文本tokenizer
        if tokenizer_name == "microsoft/deberta-v3-base":
            model_path = "downloaded_model/deberta-v3-base"
        else:
            model_path = tokenizer_name
        
        self.text_file = text_file
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length
        
        # 描述支持
        self.description_manager = None
        if description_file and os.path.exists(description_file):
            self.description_manager = ImageDescriptionManager(description_file)
        
        self.description_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.description_max_length = description_max_length
        
        # 读取数据
        self.samples = self._read_data()
        
        # 图像预处理
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
        """读取数据"""
        lines = []
        with open(self.text_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        
        assert len(lines) % 4 == 0, "MABSA 数据格式有误，每条样本应占4行。"
        
        # 聚合字典: Key = image_name
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
        
        # 转换为列表
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
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # 对齐标签
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
        
        # 加载图像
        image_tensor = self._load_image(image_path)
        
        # 编码描述
        image_name = os.path.basename(image_path)
        description_input_ids, description_attention_mask = self._get_description_encoding(image_name)
        
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            'token_type_ids': encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])).squeeze(0),
            'image_tensor': image_tensor,
            'description_input_ids': description_input_ids,
            'description_attention_mask': description_attention_mask,
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            # 容错处理：返回黑图
            return torch.zeros(3, 224, 224)
    
    def _get_description_encoding(self, image_name: str):
        """获取图像描述的编码"""
        if self.description_manager is not None:
            description = self.description_manager.get_description(image_name)
        else:
            description = "No description available."
        
        encoded = self.description_tokenizer(
            description,
            max_length=self.description_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)


class MNERDatasetDEQA(Dataset):
    """
    MNER数据集 + DEQA支持（图像描述）
    """
    
    def __init__(
        self,
        text_file: str,
        image_dir: str,
        tokenizer_name: str = "microsoft/deberta-v3-base",
        max_seq_length: int = 128,
        description_file: Optional[str] = None,
        description_max_length: int = 256
    ):
        super().__init__()
        
        # 文本tokenizer
        if tokenizer_name == "microsoft/deberta-v3-base":
            model_path = "downloaded_model/deberta-v3-base"
        else:
            model_path = tokenizer_name
        
        self.text_file = text_file
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length
                
        self.samples = self._read_data()

        # 描述支持
        self.description_manager = None
        if description_file and os.path.exists(description_file):
            self.description_manager = ImageDescriptionManager(description_file)
        
        self.description_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.description_max_length = description_max_length
        
        # 定义 type_map
        #"PER": "-1", "ORG": "0", "LOC": "1", "OTHER": "2", "MISC": "3"}
        self.type_map = {-1: 0, 0:1, 1:2, 2:3, 3:3}  # 4类
        
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _read_data(self):
        """读取数据"""
        lines = []
        with open(self.text_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        
        assert len(lines) % 4 == 0, "MNER 数据格式有误，每条样本应占4行。"
        
        for i in range(0, len(lines), 4):
            text_with_T = lines[i]
            entity_str = lines[i+1]
            entity_type_str = lines[i+2]
            image_name = lines[i+3]
            
            if not image_name.endswith(".jpg"):
                image_name += ".jpg"
            image_path = os.path.join(self.image_dir, image_name)
            
            # 先对含 $T$ 的模板文本进行标准化清洗
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
        
        # 5. 加载图像
        image_tensor = self._load_image(image_path)
        
        # 6. 编码描述
        image_name = os.path.basename(image_path)
        description_input_ids, description_attention_mask = self._get_description_encoding(image_name)
        
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            'token_type_ids': encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])).squeeze(0),
            'image_tensor': image_tensor,
            'description_input_ids': description_input_ids,
            'description_attention_mask': description_attention_mask,
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            # 容错处理：返回黑图
            return torch.zeros(3, 224, 224)
    
    def _get_description_encoding(self, image_name: str):
        """获取图像描述的编码"""
        if self.description_manager is not None:
            description = self.description_manager.get_description(image_name)
        else:
            description = "No description available."
        
        encoded = self.description_tokenizer(
            description,
            max_length=self.description_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

