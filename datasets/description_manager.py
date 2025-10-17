# datasets/description_manager.py
"""
图像描述管理器

用于加载和管理图像的文本描述（GPT-4生成或手动标注）
支持DEQA方法中的描述专家
"""

import json
import os
from typing import Dict, Optional


class ImageDescriptionManager:
    """
    图像描述管理器
    
    支持从JSONL或JSON文件加载图像描述
    格式:
    - JSONL: 每行一个JSON对象 {"image_name": "xxx.jpg", "description": "..."}
    - JSON: {"image1.jpg": "description1", "image2.jpg": "description2", ...}
    """
    
    def __init__(
        self,
        description_file: Optional[str] = None,
        default_description: str = "No description available."
    ):
        """
        Args:
            description_file: 描述文件路径 (.jsonl 或 .json)
            default_description: 默认描述（当图像没有描述时使用）
        """
        self.descriptions: Dict[str, str] = {}
        self.default_description = default_description
        
        if description_file and os.path.exists(description_file):
            self.load_descriptions(description_file)
    
    def load_descriptions(self, description_file: str):
        """
        从文件加载图像描述
        
        Args:
            description_file: 描述文件路径
        """
        if not os.path.exists(description_file):
            print(f"Warning: Description file not found: {description_file}")
            return
        
        # 根据文件扩展名判断格式
        if description_file.endswith('.jsonl'):
            self._load_jsonl(description_file)
        elif description_file.endswith('.json'):
            self._load_json(description_file)
        else:
            raise ValueError(f"Unsupported file format: {description_file}")
        
        print(f"Loaded {len(self.descriptions)} image descriptions from {description_file}")
    
    def _load_jsonl(self, file_path: str):
        """从JSONL文件加载"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    image_name = data.get('image_name', '')
                    description = data.get('description', self.default_description)
                    
                    # 标准化图像名称
                    if image_name:
                        if not image_name.endswith('.jpg'):
                            image_name += '.jpg'
                        self.descriptions[image_name] = description
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {line[:50]}... Error: {e}")
    
    def _load_json(self, file_path: str):
        """从JSON文件加载"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 标准化图像名称
        for image_name, description in data.items():
            if not image_name.endswith('.jpg'):
                image_name += '.jpg'
            self.descriptions[image_name] = description
    
    def get_description(self, image_name: str) -> str:
        """
        获取图像的描述
        
        Args:
            image_name: 图像文件名
            
        Returns:
            图像描述文本
        """
        # 标准化图像名称
        if not image_name.endswith('.jpg'):
            image_name += '.jpg'
        
        return self.descriptions.get(image_name, self.default_description)
    
    def has_description(self, image_name: str) -> bool:
        """
        检查图像是否有描述
        
        Args:
            image_name: 图像文件名
            
        Returns:
            是否有描述
        """
        if not image_name.endswith('.jpg'):
            image_name += '.jpg'
        return image_name in self.descriptions
    
    def add_description(self, image_name: str, description: str):
        """
        添加图像描述
        
        Args:
            image_name: 图像文件名
            description: 描述文本
        """
        if not image_name.endswith('.jpg'):
            image_name += '.jpg'
        self.descriptions[image_name] = description
    
    def save_descriptions(self, output_file: str, format: str = 'jsonl'):
        """
        保存图像描述到文件
        
        Args:
            output_file: 输出文件路径
            format: 输出格式 ('jsonl' 或 'json')
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if format == 'jsonl':
            with open(output_file, 'w', encoding='utf-8') as f:
                for image_name, description in self.descriptions.items():
                    data = {
                        'image_name': image_name,
                        'description': description
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
        elif format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.descriptions, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved {len(self.descriptions)} descriptions to {output_file}")
    
    def __len__(self):
        return len(self.descriptions)
    
    def __contains__(self, image_name: str):
        if not image_name.endswith('.jpg'):
            image_name += '.jpg'
        return image_name in self.descriptions


# 全局描述管理器实例（可选）
_global_description_manager: Optional[ImageDescriptionManager] = None


def get_global_description_manager() -> Optional[ImageDescriptionManager]:
    """获取全局描述管理器"""
    return _global_description_manager


def set_global_description_manager(manager: ImageDescriptionManager):
    """设置全局描述管理器"""
    global _global_description_manager
    _global_description_manager = manager


def load_global_descriptions(description_file: str):
    """加载全局描述管理器"""
    global _global_description_manager
    _global_description_manager = ImageDescriptionManager(description_file)
    return _global_description_manager

