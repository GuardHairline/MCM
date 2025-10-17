# models/model_factory.py
"""
统一的模型工厂

主要功能：
1. 简化模型创建流程
2. 统一配置管理
3. 支持不同的模型架构
4. 自动处理设备配置
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置类"""
    # 基础配置
    text_model_name: str = "microsoft/deberta-v3-base"
    image_model_name: str = "google/vit-base-patch16-224-in21k"
    hidden_dim: int = 768
    
    # 融合配置
    fusion_strategy: str = "gated"  # ["gated", "attention", "concat", "add", "adaptive"]
    num_heads: int = 8
    dropout_prob: float = 0.1
    use_dynamic_weighting: bool = False
    
    # 模式配置
    mode: str = "multimodal"  # ["text_only", "multimodal"]
    
    # 任务头配置
    use_label_embedding: bool = True
    use_hierarchical_head: bool = False
    
    # 持续学习配置
    cl_method: Optional[str] = None  # ["ewc", "replay", "lwf", "si", "mas", "gem", "tam_cl", "moe", "clap4clip"]
    
    # MOE配置
    num_experts: int = 4
    use_ddas: bool = False
    
    # 其他配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'text_model_name': self.text_model_name,
            'image_model_name': self.image_model_name,
            'hidden_dim': self.hidden_dim,
            'fusion_strategy': self.fusion_strategy,
            'num_heads': self.num_heads,
            'dropout_prob': self.dropout_prob,
            'use_dynamic_weighting': self.use_dynamic_weighting,
            'mode': self.mode,
            'use_label_embedding': self.use_label_embedding,
            'use_hierarchical_head': self.use_hierarchical_head,
            'cl_method': self.cl_method,
            'num_experts': self.num_experts,
            'use_ddas': self.use_ddas,
            'device': self.device
        }


class ModelFactory:
    """
    模型工厂类
    
    用于创建不同类型的模型，统一接口
    """
    
    @staticmethod
    def create_base_model(config: ModelConfig):
        """
        创建基础多模态模型
        
        Args:
            config: 模型配置
        
        Returns:
            base_model: 基础编码器模型
        """
        # 优先使用增强版
        try:
            from models.base_model_enhanced import BaseMultimodalModelEnhanced
            
            base_model = BaseMultimodalModelEnhanced(
                text_model_name=config.text_model_name,
                image_model_name=config.image_model_name,
                hidden_dim=config.hidden_dim,
                multimodal_fusion=config.fusion_strategy,
                num_heads=config.num_heads,
                mode=config.mode,
                dropout_prob=config.dropout_prob,
                use_dynamic_weighting=config.use_dynamic_weighting
            )
            
            logger.info(f"Created BaseMultimodalModelEnhanced with fusion={config.fusion_strategy}")
            
        except Exception as e:
            logger.warning(f"Failed to create enhanced model: {e}, falling back to base model")
            
            from models.base_model import BaseMultimodalModel
            
            base_model = BaseMultimodalModel(
                text_model_name=config.text_model_name,
                image_model_name=config.image_model_name,
                hidden_dim=config.hidden_dim,
                multimodal_fusion=config.fusion_strategy,
                num_heads=config.num_heads,
                mode=config.mode,
                dropout_prob=config.dropout_prob
            )
            
            logger.info(f"Created BaseMultimodalModel with fusion={config.fusion_strategy}")
        
        return base_model
    
    @staticmethod
    def create_full_model(config: ModelConfig, task_name: str, num_labels: int,
                         label_embedding_manager=None):
        """
        创建完整模型（包含任务头）
        
        Args:
            config: 模型配置
            task_name: 任务名称
            num_labels: 标签数量
            label_embedding_manager: 标签嵌入管理器
        
        Returns:
            model: 完整模型
        """
        from modules.train_utils import Full_Model
        
        # 创建基础模型
        base_model = ModelFactory.create_base_model(config)
        
        # 创建完整模型
        full_model = Full_Model(
            base_model=base_model,
            num_labels=num_labels,
            task_name=task_name,
            label_embedding_manager=label_embedding_manager,
            device=config.device
        )
        
        logger.info(f"Created Full_Model for task '{task_name}' with {num_labels} labels")
        
        return full_model
    
    @staticmethod
    def create_cl_model(config: ModelConfig, task_name: str, num_labels: int,
                       label_embedding_manager=None, session_name: str = None):
        """
        创建持续学习模型
        
        Args:
            config: 模型配置
            task_name: 任务名称
            num_labels: 标签数量
            label_embedding_manager: 标签嵌入管理器
            session_name: 会话名称
        
        Returns:
            model: 持续学习模型
        """
        if config.cl_method == "tam_cl":
            from continual.tam_cl import TamCLModel
            
            model = TamCLModel(
                text_model_name=config.text_model_name,
                image_model_name=config.image_model_name,
                num_labels=num_labels,
                task_name=task_name,
                fusion_strategy=config.fusion_strategy
            )
            
            logger.info(f"Created TAM-CL model for task '{task_name}'")
            
        elif config.cl_method == "moe" or config.cl_method == "moeadapter":
            from continual.moe_adapters.moe_model_wrapper import MoeAdapterWrapper
            from continual.moe_adapters.ddas_router import DDASRouter
            
            base_model = ModelFactory.create_base_model(config)
            
            model = MoeAdapterWrapper(
                base_model=base_model,
                num_experts=config.num_experts,
                use_ddas=config.use_ddas
            )
            
            logger.info(f"Created MOE model with {config.num_experts} experts, DDAS={config.use_ddas}")
            
        elif config.cl_method == "clap4clip":
            from continual.clap4clip.clap4clip import CLAP4CLIP
            
            model = CLAP4CLIP(
                text_model_name=config.text_model_name,
                image_model_name=config.image_model_name,
                num_labels=num_labels,
                task_name=task_name
            )
            
            logger.info(f"Created CLAP4CLIP model for task '{task_name}'")
            
        else:
            # 使用标准Full_Model
            model = ModelFactory.create_full_model(
                config, task_name, num_labels, label_embedding_manager
            )
        
        # 移动到设备
        model = model.to(config.device)
        
        return model
    
    @staticmethod
    def from_args(args, task_name: str = None, num_labels: int = None,
                  label_embedding_manager=None):
        """
        从argparse参数创建模型
        
        Args:
            args: argparse.Namespace
            task_name: 任务名称（如果不在args中）
            num_labels: 标签数量（如果不在args中）
            label_embedding_manager: 标签嵌入管理器
        
        Returns:
            model: 模型实例
        """
        # 从args构建配置
        config = ModelConfig(
            text_model_name=getattr(args, 'text_model_name', 'microsoft/deberta-v3-base'),
            image_model_name=getattr(args, 'image_model_name', 'google/vit-base-patch16-224-in21k'),
            hidden_dim=getattr(args, 'hidden_dim', 768),
            fusion_strategy=getattr(args, 'fusion_strategy', 'gated'),
            num_heads=getattr(args, 'num_heads', 8),
            dropout_prob=getattr(args, 'dropout_prob', 0.1),
            use_dynamic_weighting=getattr(args, 'use_dynamic_weighting', False),
            mode=getattr(args, 'mode', 'multimodal'),
            use_label_embedding=getattr(args, 'use_label_embedding', True),
            use_hierarchical_head=getattr(args, 'use_hierarchical_head', False),
            cl_method=None,  # 从特定标志推断
            num_experts=getattr(args, 'num_experts', 4),
            use_ddas=getattr(args, 'ddas', False),
            device=getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # 推断CL方法
        if getattr(args, 'tam_cl', False):
            config.cl_method = 'tam_cl'
        elif getattr(args, 'moe_adapters', False):
            config.cl_method = 'moe'
        elif getattr(args, 'clap4clip', False):
            config.cl_method = 'clap4clip'
        
        # 获取任务信息
        if task_name is None:
            task_name = getattr(args, 'task_name', 'unknown')
        if num_labels is None:
            num_labels = getattr(args, 'num_labels', 3)
        
        session_name = getattr(args, 'session_name', task_name)
        
        # 创建模型
        model = ModelFactory.create_cl_model(
            config, task_name, num_labels, 
            label_embedding_manager, session_name
        )
        
        logger.info(f"Created model from args: task={task_name}, cl_method={config.cl_method}")
        
        return model, config


def create_model_simple(task_name: str, num_labels: int, 
                       fusion_strategy: str = "gated",
                       use_label_embedding: bool = True,
                       device: str = None) -> nn.Module:
    """
    简化的模型创建接口
    
    Args:
        task_name: 任务名称
        num_labels: 标签数量
        fusion_strategy: 融合策略
        use_label_embedding: 是否使用标签嵌入
        device: 设备
    
    Returns:
        model: 模型实例
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = ModelConfig(
        fusion_strategy=fusion_strategy,
        use_label_embedding=use_label_embedding,
        device=device
    )
    
    # 创建标签嵌入管理器（如果需要）
    label_embedding_manager = None
    if use_label_embedding:
        try:
            from continual.label_embedding_manager import LabelEmbeddingManager
            label_embedding_manager = LabelEmbeddingManager()
        except:
            logger.warning("Failed to create LabelEmbeddingManager")
    
    model = ModelFactory.create_full_model(
        config, task_name, num_labels, label_embedding_manager
    )
    
    return model.to(device)

