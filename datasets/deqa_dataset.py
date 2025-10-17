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
        
        assert len(lines) % 4 == 0, "MATE 数据格式有误，每条样本应占4行。"
        
        for i in range(0, len(lines), 4):
            text_with_T = lines[i]
            aspect_term = lines[i+1]
            _ = lines[i+2]  # sentiment
            image_name = lines[i+3]
            
            if not image_name.endswith(".jpg"):
                image_name += ".jpg"
            image_path = os.path.join(self.image_dir, image_name)
            
            self.samples.append((text_with_T, aspect_term, image_name, image_path))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text_with_T, aspect_term, image_name, image_path = self.samples[idx]
        
        # 1. 替换 $T$
        if "$T$" in text_with_T:
            T_position = text_with_T.index("$T$")
            replaced_text = text_with_T.replace("$T$", aspect_term)
            start_pos = T_position
            end_pos = start_pos + len(aspect_term) - 1
        else:
            replaced_text = text_with_T
            start_pos = -1
            end_pos = -1
        
        # 2. 构造char级别标签
        char_label = [0] * len(replaced_text)
        if 0 <= start_pos < len(replaced_text):
            char_label[start_pos] = 1  # B
            for c in range(start_pos+1, min(end_pos+1, len(replaced_text))):
                char_label[c] = 2  # I
        
        # 3. Tokenize
        encoded = self.tokenizer(
            replaced_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # 4. 对齐标签
        offsets = encoded.pop('offset_mapping').squeeze(0)
        token_labels = []
        for offset in offsets:
            if offset[0] == 0 and offset[1] == 0:
                token_labels.append(-100)
            else:
                start_char, end_char = offset
                if start_char < len(char_label):
                    token_labels.append(char_label[start_char])
                else:
                    token_labels.append(0)
        
        # 5. 加载图像
        image_tensor = self._load_image(image_path)
        
        # 6. 编码描述
        description_input_ids, description_attention_mask = self._get_description_encoding(image_name)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'token_type_ids': encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])).squeeze(0),
            'image_tensor': image_tensor,
            'description_input_ids': description_input_ids,
            'description_attention_mask': description_attention_mask,
            'labels': torch.tensor(token_labels, dtype=torch.long)
        }
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """加载和预处理图像"""
        try:
            if not os.path.exists(image_path):
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            else:
                image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            return self.image_transform(image)
    
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

