# utils/description_generator.py
"""
图像描述生成工具

使用GPT-4或其他多模态大语言模型为图像生成文本描述
供DEQA方法使用

注意：此工具为可选工具，需要OpenAI API密钥
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
from tqdm import tqdm


class ImageDescriptionGenerator:
    """
    图像描述生成器
    
    支持：
    1. GPT-4 Vision (需要OpenAI API)
    2. BLIP (本地模型，无需API)
    3. LLaVA (本地模型，无需API)
    """
    
    def __init__(
        self,
        model_type: str = 'blip',
        api_key: Optional[str] = None,
        max_length: int = 100
    ):
        """
        Args:
            model_type: 模型类型 ('gpt4', 'blip', 'llava')
            api_key: OpenAI API密钥（仅GPT-4需要）
            max_length: 最大描述长度
        """
        self.model_type = model_type
        self.api_key = api_key
        self.max_length = max_length
        
        if model_type == 'gpt4':
            self._init_gpt4()
        elif model_type == 'blip':
            self._init_blip()
        elif model_type == 'llava':
            self._init_llava()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _init_gpt4(self):
        """初始化GPT-4 Vision"""
        try:
            import openai
            if not self.api_key:
                self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key not provided")
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
            print("GPT-4 Vision initialized")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def _init_blip(self):
        """初始化BLIP模型"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            
            print("Loading BLIP model...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # 使用GPU加速
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            print(f"BLIP model loaded on {self.device}")
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
    
    def _init_llava(self):
        """初始化LLaVA模型"""
        raise NotImplementedError("LLaVA support coming soon")
    
    def generate_description_gpt4(self, image_path: str) -> str:
        """使用GPT-4 Vision生成描述"""
        # 读取图像并编码为base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please provide a detailed description of this image in one or two sentences."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_length
            )
            
            description = response.choices[0].message.content
            return description.strip()
        except Exception as e:
            print(f"Error generating description for {image_path}: {e}")
            return "No description available."
    
    def generate_description_blip(self, image_path: str) -> str:
        """使用BLIP生成描述"""
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 处理图像
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # 生成描述
            out = self.model.generate(**inputs, max_length=self.max_length)
            description = self.processor.decode(out[0], skip_special_tokens=True)
            
            return description.strip()
        except Exception as e:
            print(f"Error generating description for {image_path}: {e}")
            return "No description available."
    
    def generate_description(self, image_path: str) -> str:
        """生成图像描述"""
        if self.model_type == 'gpt4':
            return self.generate_description_gpt4(image_path)
        elif self.model_type == 'blip':
            return self.generate_description_blip(image_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def generate_descriptions_batch(
        self,
        image_dir: str,
        output_file: str,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png']
    ):
        """
        批量生成图像描述
        
        Args:
            image_dir: 图像目录
            output_file: 输出文件路径 (.jsonl)
            image_extensions: 图像文件扩展名列表
        """
        # 获取所有图像文件
        image_dir = Path(image_dir)
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
        
        print(f"Found {len(image_files)} images in {image_dir}")
        
        # 创建输出目录
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成描述
        descriptions = []
        with open(output_file, 'w', encoding='utf-8') as f:
            for image_file in tqdm(image_files, desc="Generating descriptions"):
                description = self.generate_description(str(image_file))
                
                data = {
                    'image_name': image_file.name,
                    'description': description
                }
                
                # 写入文件
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                f.flush()  # 立即写入
                
                descriptions.append(data)
        
        print(f"Saved {len(descriptions)} descriptions to {output_file}")
        return descriptions


def main():
    """
    示例：为Twitter2015数据集生成图像描述
    
    用法:
    python utils/description_generator.py \
        --image_dir data/twitter2015_images \
        --output_file data/twitter2015_descriptions.jsonl \
        --model_type blip
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate image descriptions for DEQA")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing images")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output file path (.jsonl)")
    parser.add_argument("--model_type", type=str, default='blip',
                       choices=['gpt4', 'blip', 'llava'],
                       help="Model type for description generation")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key (for GPT-4)")
    parser.add_argument("--max_length", type=int, default=100,
                       help="Maximum description length")
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = ImageDescriptionGenerator(
        model_type=args.model_type,
        api_key=args.api_key,
        max_length=args.max_length
    )
    
    # 批量生成描述
    generator.generate_descriptions_batch(
        image_dir=args.image_dir,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()

