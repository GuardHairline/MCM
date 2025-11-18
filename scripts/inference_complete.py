#!/usr/bin/env python3
"""
完整的推理脚本

基于训练流程（train_with_zero_shot.py）设计的推理接口

支持的任务：
- MATE: 多模态方面术语提取（序列标注）
- MNER: 多模态命名实体识别（序列标注）
- MABSA: 多模态方面情感分析（序列标注）
- MASC: 多模态方面情感分类（句子分类）

保存的模型文件：
1. {base_name}.pt - 完整模型 state_dict（包含 base_model 和所有任务头）
2. {base_name}_task_heads.pt - 任务头单独保存（可选）
3. train_info_{base_name}.json - 训练信息和历史指标
4. label_embedding_{base_name}.pt - 标签嵌入（如果使用）

使用示例：
    # MASC（句子级分类）
    predictor = MultimodalInference(
        model_path="checkpoints/twitter2015_none_t2m_seq1.pt",
        train_info_path="checkpoints/train_info_twitter2015_none_t2m_seq1.json",
        task_name="masc",
        session_name="twitter2015_masc_multimodal"
    )
    result = predictor.predict_sentence(
        text="The $T$ is great",
        aspect="food",
        image_path="data/twitter2015/images/12345.jpg"
    )
    
    # MATE（序列标注）
    predictor = MultimodalInference(
        model_path="checkpoints/twitter2015_none_t2m_seq1.pt",
        train_info_path="checkpoints/train_info_twitter2015_none_t2m_seq1.json",
        task_name="mate",
        session_name="twitter2015_mate_multimodal"
    )
    result = predictor.predict_sequence(
        text="The food is great but service sucks",
        image_path="data/twitter2015/images/12345.jpg"
    )
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from PIL import Image
import json
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoTokenizer
from torchvision import transforms

from models.base_model import BaseMultimodalModel
from modules.train_utils import Full_Model
from models.task_heads.mate_head import MATEHead
from models.task_heads.mner_head import MNERHead  
from models.task_heads.mabsa_head import MABSAHead
from models.task_heads.sent_label_attn import LabelAttentionSentHead
from continual.label_config import get_label_manager


class MultimodalInference:
    """
    多模态模型推理器
    
    自动从训练信息中推断模型配置，支持多任务推理
    """
    
    def __init__(self,
                 model_path: str,
                 train_info_path: str,
                 task_name: str,
                 session_name: str = None,
                 device: str = None):
        """
        初始化推理器
        
        Args:
            model_path: 模型文件路径（.pt）
            train_info_path: 训练信息文件路径（.json）
            task_name: 任务名称 (mate, mner, mabsa, masc)
            session_name: 会话名称（如果None，自动查找）
            device: 设备（如果None，自动选择）
        """
        self.task_name = task_name.lower()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载训练信息
        self.train_info = self._load_train_info(train_info_path)
        
        # 确定session_name
        if session_name is None:
            session_name = self._find_session_for_task(self.task_name)
            if session_name is None:
                raise ValueError(f"Could not find session for task '{self.task_name}' in train_info")
        
        self.session_name = session_name
        
        # 获取任务配置
        self.task_config = self._get_task_config()
        self.num_labels = self.task_config['num_labels']
        self.mode = self.task_config['mode']
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 初始化tokenizer和图像transform
        self._init_tokenizer()
        self._init_image_transform()
        
        # 获取标签管理器
        self.label_manager = get_label_manager()
        self.task_info = self.label_manager.get_task_config(self.task_name)
        
        print(f"\n{'='*80}")
        print(f"多模态推理器初始化成功")
        print(f"{'='*80}")
        print(f"任务: {self.task_name.upper()}")
        print(f"会话: {self.session_name}")
        print(f"标签数: {self.num_labels}")
        print(f"模式: {self.mode}")
        print(f"设备: {self.device}")
        print(f"{'='*80}\n")
    
    def _load_train_info(self, path: str) -> Dict:
        """加载训练信息"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _find_session_for_task(self, task_name: str) -> Optional[str]:
        """从训练信息中查找任务对应的session"""
        # 新格式：sessions列表
        if 'sessions' in self.train_info:
            for session in self.train_info['sessions']:
                if session['task_name'].lower() == task_name:
                    return session['session_name']
        
        # 旧格式：tasks列表
        if 'tasks' in self.train_info:
            for i, task in enumerate(self.train_info['tasks']):
                if task.lower() == task_name:
                    # 尝试构造session_name
                    dataset = self.train_info.get('dataset', 'twitter2015')
                    return f"{dataset}_{task_name}_multimodal"
        
        return None
    
    def _get_task_config(self) -> Dict:
        """获取任务配置"""
        # 从sessions中查找
        if 'sessions' in self.train_info:
            for session in self.train_info['sessions']:
                if session['session_name'] == self.session_name:
                    return session
        
        # 默认配置
        label_counts = {
            'mate': 3,
            'mner': 9,
            'mabsa': 7,
            'masc': 3
        }
        
        return {
            'task_name': self.task_name,
            'session_name': self.session_name,
            'num_labels': label_counts.get(self.task_name, 3),
            'mode': 'multimodal'
        }
    
    def _load_model(self, model_path: str) -> Full_Model:
        """加载完整模型"""
        print(f"加载模型: {model_path}")
        
        # 创建base_model
        base_model = BaseMultimodalModel(
            text_model_name="microsoft/deberta-v3-base",
            image_model_name="google/vit-base-patch16-224-in21k",
            mode=self.mode
        )
        
        # 创建任务头
        task_head = self._create_task_head(base_model)
        
        # 创建完整模型
        full_model = Full_Model(
            base_model=base_model,
            head=task_head,
            device=self.device
        )
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 处理可能的state_dict格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 加载state_dict
        try:
            full_model.load_state_dict(state_dict, strict=False)
            print("✓ 模型权重加载成功")
        except Exception as e:
            print(f"⚠️ 加载权重时出现警告: {e}")
            print("  尝试宽松加载...")
            missing_keys, unexpected_keys = full_model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"  缺失的键: {len(missing_keys)}")
            if unexpected_keys:
                print(f"  未预期的键: {len(unexpected_keys)}")
        
        full_model.to(self.device)
        return full_model
    
    def _create_task_head(self, base_model) -> torch.nn.Module:
        """根据任务类型创建任务头"""
        input_dim = base_model.text_hidden_size
        
        # 检查是否使用CRF
        use_crf = self.task_config.get('use_crf', False)
        if isinstance(use_crf, int):
            use_crf = bool(use_crf)
        
        if self.task_name == 'mate':
            return MATEHead(
                input_dim=input_dim,
                num_labels=self.num_labels,
                use_crf=use_crf
            )
        elif self.task_name == 'mner':
            return MNERHead(
                input_dim=input_dim,
                num_labels=self.num_labels,
                use_crf=use_crf
            )
        elif self.task_name == 'mabsa':
            return MABSAHead(
                input_dim=input_dim,
                num_labels=self.num_labels,
                use_crf=use_crf
            )
        elif self.task_name == 'masc':
            # MASC是句子级分类，使用LabelAttentionSentHead
            return LabelAttentionSentHead(
                input_dim=input_dim,
                num_labels=self.num_labels,
                label_emb=None  # 推理时不需要label embedding
            )
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")
    
    def _init_tokenizer(self):
        """初始化tokenizer"""
        model_path = "downloaded_model/deberta-v3-base"
        if not os.path.exists(model_path):
            model_path = "microsoft/deberta-v3-base"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = 128
    
    def _init_image_transform(self):
        """初始化图像转换"""
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """加载和处理图像"""
        try:
            with Image.open(image_path).convert('RGB') as img:
                img_tensor = self.image_transform(img)
            return img_tensor.unsqueeze(0)  # (1, 3, 224, 224)
        except Exception as e:
            print(f"⚠️ 图像加载失败: {e}, 使用零张量")
            return torch.zeros((1, 3, 224, 224))
    
    def predict_sentence(self,
                        text: str,
                        aspect: str,
                        image_path: str) -> Dict[str, Any]:
        """
        句子级分类预测（用于MASC）
        
        Args:
            text: 输入文本（可能包含$T$占位符）
            aspect: 方面词
            image_path: 图像路径
        
        Returns:
            {
                'text': str,
                'aspect': str,
                'sentiment': int,  # -1, 0, 1
                'sentiment_name': str,  # 'negative', 'neutral', 'positive'
                'probabilities': dict  # {label: prob}
            }
        """
        if self.task_name != 'masc':
            raise ValueError(f"predict_sentence only supports MASC task, got {self.task_name}")
        
        # 替换$T$
        if "$T$" in text:
            processed_text = text.replace("$T$", f"<asp> {aspect} </asp>")
        else:
            processed_text = text + f" <asp> {aspect} </asp>"
        
        # Tokenize
        encoding = self.tokenizer(
            processed_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        
        # Load image
        image_tensor = self._load_image(image_path).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, token_type_ids, image_tensor)
            probs = F.softmax(logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()
        
        # 映射到情感值
        label_to_sentiment = {0: -1, 1: 0, 2: 1}
        sentiment = label_to_sentiment[pred_label]
        
        sentiment_names = {-1: 'negative', 0: 'neutral', 1: 'positive'}
        
        return {
            'text': text,
            'aspect': aspect,
            'sentiment': sentiment,
            'sentiment_name': sentiment_names[sentiment],
            'probabilities': {
                'negative': probs[0, 0].item(),
                'neutral': probs[0, 1].item(),
                'positive': probs[0, 2].item()
            },
            'confidence': probs[0, pred_label].item()
        }
    
    def predict_sequence(self,
                        text: str,
                        image_path: str,
                        return_tokens: bool = True) -> Dict[str, Any]:
        """
        序列标注预测（用于MATE, MNER, MABSA）
        
        Args:
            text: 输入文本
            image_path: 图像路径
            return_tokens: 是否返回token级别的预测
        
        Returns:
            {
                'text': str,
                'entities': list,  # [(start, end, label, text), ...]
                'token_predictions': list  # [(token, label), ...] if return_tokens
            }
        """
        if self.task_name not in ['mate', 'mner', 'mabsa']:
            raise ValueError(f"predict_sequence only supports MATE/MNER/MABSA, got {self.task_name}")
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        offsets = encoding['offset_mapping'][0]
        
        # Load image
        image_tensor = self._load_image(image_path).to(self.device)
        
        # Predict
        with torch.no_grad():
            # Check if using CRF
            if hasattr(self.model.head, 'crf') and self.model.head.crf is not None:
                # CRF decode
                logits = self.model.head(
                    self.model.base_model(input_ids, attention_mask, token_type_ids, image_tensor, return_sequence=True),
                    labels=None
                )
                # CRF decode returns best path
                # 需要手动decode
                predictions = self._crf_decode(logits, attention_mask)
            else:
                # Standard argmax
                logits = self.model(input_ids, attention_mask, token_type_ids, image_tensor)
                predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        
        # Decode tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Extract entities
        entities = self._extract_entities(predictions, tokens, offsets, text)
        
        result = {
            'text': text,
            'entities': entities
        }
        
        if return_tokens:
            token_predictions = []
            for i, (token, pred) in enumerate(zip(tokens, predictions)):
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    label_name = self.task_info.label_names[pred] if self.task_info else str(pred)
                    token_predictions.append((token, label_name))
            result['token_predictions'] = token_predictions
        
        return result
    
    def _crf_decode(self, logits: torch.Tensor, mask: torch.Tensor) -> List[int]:
        """CRF解码"""
        # 使用CRF的decode方法
        if hasattr(self.model.head, 'crf'):
            # 去除[CLS]和[SEP]
            # 找到有效范围
            valid_indices = (mask[0] == 1).nonzero(as_tuple=True)[0]
            if len(valid_indices) > 1:
                start_idx = valid_indices[0].item() + 1  # 跳过[CLS]
                end_idx = valid_indices[-1].item()  # 跳过[SEP]
                
                if end_idx > start_idx:
                    cropped_logits = logits[:, start_idx:end_idx, :]
                    cropped_mask = torch.ones(1, end_idx - start_idx, dtype=torch.bool, device=logits.device)
                    
                    decoded = self.model.head.crf.decode(cropped_logits, mask=cropped_mask)[0]
                    
                    # 填充回完整序列
                    full_predictions = [0] * logits.size(1)
                    for i, pred in enumerate(decoded):
                        full_predictions[start_idx + i] = pred
                    
                    return full_predictions
        
        # Fallback: argmax
        return torch.argmax(logits, dim=-1)[0].cpu().numpy().tolist()
    
    def _extract_entities(self,
                         predictions: List[int],
                         tokens: List[str],
                         offsets: List[Tuple[int, int]],
                         text: str) -> List[Tuple[int, int, str, str]]:
        """从BIO标签提取实体"""
        entities = []
        current_entity = None
        
        for i, (pred, token, offset) in enumerate(zip(predictions, tokens, offsets)):
            # 跳过特殊token
            if token in ['[CLS]', '[SEP]', '[PAD]'] or offset[0] == offset[1]:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            label_name = self.task_info.label_names[pred] if self.task_info else str(pred)
            
            if label_name == 'O':
                # Outside
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            elif label_name.startswith('B-') or (self.task_name == 'mate' and pred == 1):
                # Begin
                if current_entity:
                    entities.append(current_entity)
                
                # Extract entity type
                entity_type = label_name[2:] if '-' in label_name else 'ENTITY'
                start_char = offset[0]
                current_entity = [start_char, offset[1], entity_type, text[start_char:offset[1]]]
            elif label_name.startswith('I-') or (self.task_name == 'mate' and pred == 2):
                # Inside
                if current_entity:
                    # Extend current entity
                    current_entity[1] = offset[1]
                    current_entity[3] = text[current_entity[0]:offset[1]]
        
        # Close last entity
        if current_entity:
            entities.append(current_entity)
        
        return [tuple(e) for e in entities]


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description="多模态模型推理")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--train_info_path", type=str, required=True, help="训练信息路径")
    parser.add_argument("--task", type=str, required=True, choices=['mate', 'mner', 'mabsa', 'masc'])
    parser.add_argument("--text", type=str, required=True, help="输入文本")
    parser.add_argument("--image", type=str, required=True, help="图像路径")
    parser.add_argument("--aspect", type=str, default=None, help="方面词（MASC任务必需）")
    parser.add_argument("--session", type=str, default=None, help="会话名称（可选）")
    
    args = parser.parse_args()
    
    # 创建推理器
    predictor = MultimodalInference(
        model_path=args.model_path,
        train_info_path=args.train_info_path,
        task_name=args.task,
        session_name=args.session
    )
    
    # 执行预测
    if args.task == 'masc':
        if not args.aspect:
            raise ValueError("MASC任务需要提供 --aspect 参数")
        result = predictor.predict_sentence(args.text, args.aspect, args.image)
        
        print("\n" + "="*80)
        print("预测结果（MASC - 句子级分类）")
        print("="*80)
        print(f"文本: {result['text']}")
        print(f"方面词: {result['aspect']}")
        print(f"情感: {result['sentiment_name']} ({result['sentiment']})")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"\n概率分布:")
        for sent, prob in result['probabilities'].items():
            print(f"  {sent}: {prob:.4f}")
        print("="*80)
        
    else:
        result = predictor.predict_sequence(args.text, args.image)
        
        print("\n" + "="*80)
        print(f"预测结果（{args.task.upper()} - 序列标注）")
        print("="*80)
        print(f"文本: {result['text']}")
        print(f"\n识别的实体:")
        if result['entities']:
            for start, end, label, entity_text in result['entities']:
                print(f"  [{start}:{end}] {label}: {entity_text}")
        else:
            print("  (无)")
        
        if 'token_predictions' in result:
            print(f"\nToken级别预测:")
            for token, label in result['token_predictions'][:20]:  # 只显示前20个
                print(f"  {token:>15} -> {label}")
        print("="*80)


if __name__ == "__main__":
    main()

