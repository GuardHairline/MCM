# models/deqa_expert_model.py
"""
DEQA专家模型 - 集成到现有的多模态模型框架

该模块将DEQA的多专家集成思想融入到现有的持续学习框架中
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from transformers import AutoModel, AutoTokenizer

from models.base_model import BaseMultimodalModel
from models.task_head_manager import TaskHeadManager
from continual.deqa import DEQACL, DEQAEnsemble


class DEQAMultimodalModel(nn.Module):
    """
    DEQA多模态模型 - 基于BaseMultimodalModel的DEQA扩展
    
    核心特性:
    1. 支持文本、描述和CLIP图像特征的多专家集成
    2. 为每个任务维护独立的专家集成
    3. 支持持续学习和知识蒸馏
    
    Args:
        text_model_name: 文本编码器名称
        image_model_name: 图像编码器名称
        fusion_strategy: 融合策略
        num_heads: 注意力头数
        mode: 模式 ('text_only', 'multimodal')
        hidden_dim: 隐藏层维度
        dropout_prob: Dropout概率
        use_description: 是否使用描述专家
        use_clip: 是否使用CLIP专家
        ensemble_method: 集成方法 ('vote', 'weighted', 'learned')
        freeze_old_experts: 是否冻结旧任务专家
        distill_weight: 蒸馏损失权重
    """
    def __init__(
        self,
        text_model_name: str = "microsoft/deberta-v3-base",
        image_model_name: str = "google/vit-base-patch16-224-in21k",
        fusion_strategy: str = "concat",
        num_heads: int = 8,
        mode: str = "multimodal",
        hidden_dim: int = 768,
        dropout_prob: float = 0.1,
        use_description: bool = True,
        use_clip: bool = True,
        ensemble_method: str = 'weighted',
        freeze_old_experts: bool = True,
        distill_weight: float = 0.5,
        **kwargs
    ):
        super().__init__()
        
        # 基础多模态模型
        self.base_model = BaseMultimodalModel(
            text_model_name=text_model_name,
            image_model_name=image_model_name,
            hidden_dim=hidden_dim,
            multimodal_fusion=fusion_strategy,
            num_heads=num_heads,
            mode=mode,
            dropout_prob=dropout_prob
        )
        
        # 描述编码器（与文本编码器共享或独立）
        self.use_description = use_description
        if use_description:
            # 使用独立的描述编码器
            model_path = "downloaded_model/deberta-v3-base" if text_model_name == "microsoft/deberta-v3-base" else text_model_name
            self.description_encoder = AutoModel.from_pretrained(model_path)
        else:
            self.description_encoder = None
        
        # DEQA持续学习组件
        self.deqa_cl = DEQACL(
            text_dim=self.base_model.text_hidden_size,
            use_description=use_description,
            use_clip=use_clip,
            ensemble_method=ensemble_method,
            distill_weight=distill_weight,
            freeze_old_experts=freeze_old_experts,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob
        )
        
        # 任务头管理器（用于管理不同任务的输出层）
        self.head_manager = TaskHeadManager(
            hidden_dim=self.base_model.fusion_output_dim
        )
        
        # 当前激活的任务
        self.current_task = None
        self.current_session = None
        
        # 模式设置
        self.mode = mode
        self.use_clip = use_clip
        
    def add_task(self, task_name: str, session_name: str, num_labels: int, args):
        """
        为新任务添加DEQA专家集成
        
        Args:
            task_name: 任务名称 (mate, masc, mner, mabsa等)
            session_name: 会话名称 (唯一标识符)
            num_labels: 标签数量
            args: 训练参数
        """
        # ✓ 修改：添加DEQA专家集成（不包含分类头，输出特征）
        self.deqa_cl.add_task(
            task_name=session_name,
            num_labels=None,  # ✓ 不需要，因为use_head=False
            use_head=False  # ✓ 重要：专家只输出特征，不输出logits
        )
        
        # 记录当前任务
        self.current_task = task_name
        self.current_session = session_name
        
        # ✓ 重要：为任务创建任务头（现在会被使用！）
        use_label_embedding = getattr(args, 'use_label_embedding', False)
        self.head_manager.create_and_register_head(
            session_name, task_name, args, use_label_embedding
        )
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        编码文本特征（仅文本，不含图像）
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs
            return_sequence: 是否返回序列特征
            
        Returns:
            文本特征
        """
        if token_type_ids is not None:
            text_outputs = self.base_model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            text_outputs = self.base_model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        if return_sequence:
            return text_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        else:
            return text_outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
    
    def encode_description(
        self,
        description_input_ids: torch.Tensor,
        description_attention_mask: torch.Tensor,
        return_sequence: bool = False
    ) -> Optional[torch.Tensor]:
        """
        编码描述特征
        
        Args:
            description_input_ids: 描述token IDs
            description_attention_mask: 描述注意力掩码
            return_sequence: 是否返回序列特征
            
        Returns:
            描述特征
        """
        if not self.use_description or self.description_encoder is None:
            return None
        
        desc_outputs = self.description_encoder(
            input_ids=description_input_ids,
            attention_mask=description_attention_mask
        )
        
        if return_sequence:
            return desc_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        else:
            return desc_outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        image_tensor: Optional[torch.Tensor] = None,
        description_input_ids: Optional[torch.Tensor] = None,
        description_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        task_name: Optional[str] = None,
        session_name: Optional[str] = None,
        return_expert_logits: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        DEQA前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs
            image_tensor: 图像张量
            description_input_ids: 描述token IDs
            description_attention_mask: 描述注意力掩码
            labels: 标签
            task_name: 任务名称
            session_name: 会话名称
            return_expert_logits: 是否返回专家logits
            
        Returns:
            输出字典，包含logits、loss等
        """
        # 确定使用的任务
        if session_name is None:
            session_name = self.current_session
        if task_name is None:
            task_name = self.current_task
        
        if session_name is None:
            raise ValueError("session_name must be provided or set via add_task")
        
        # 判断是否为序列标注任务
        sequence_tasks = {"mate", "mner", "mabsa", "ate", "ner", "absa"}
        is_sequence_task = task_name in sequence_tasks
        
        # === Expert 1: Text-only features ===
        text_features = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_sequence=is_sequence_task
        )
        
        # === Expert 2: Description features ===
        description_features = None
        if self.use_description and description_input_ids is not None:
            description_features = self.encode_description(
                description_input_ids=description_input_ids,
                description_attention_mask=description_attention_mask,
                return_sequence=is_sequence_task
            )
        
        # === Expert 3: CLIP features (multimodal fusion) ===
        clip_features = None
        if self.use_clip and self.mode != "text_only" and image_tensor is not None:
            clip_features = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                image_tensor=image_tensor,
                return_sequence=is_sequence_task
            )
        
        # === DEQA多专家集成 → 融合特征 ===
        # ✓ 重要修改：获取融合特征，而不是logits
        if return_expert_logits:
            ensemble_features, expert_features_dict = self.deqa_cl(
                task_name=session_name,
                text_features=text_features,
                description_features=description_features,
                clip_features=clip_features,
                return_expert_logits=True,
                return_features=True  # ✓ 返回特征
            )
        else:
            ensemble_features = self.deqa_cl(
                task_name=session_name,
                text_features=text_features,
                description_features=description_features,
                clip_features=clip_features,
                return_features=True  # ✓ 返回特征
            )
            expert_features_dict = None
        
        # ✓ 新增：使用head_manager输出logits（与框架统一！）
        if hasattr(self, 'head') and self.head is not None:
            logits = self.head(ensemble_features)
        else:
            raise ValueError(f"No head set for session {session_name}")
        
        # 计算损失
        loss = None
        if labels is not None:
            # 根据任务类型计算损失
            if is_sequence_task:
                # 序列标注任务 - 使用token级损失
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                # logits: (batch_size, seq_len, num_labels)
                # labels: (batch_size, seq_len)
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
            else:
                # 句级分类任务
                loss_fct = nn.CrossEntropyLoss()
                # logits: (batch_size, num_labels)
                # labels: (batch_size,)
                loss = loss_fct(logits, labels)
            
            # 添加蒸馏损失
            distill_loss = self.deqa_cl.compute_distillation_loss(
                task_name=session_name,
                text_features=text_features,
                description_features=description_features,
                clip_features=clip_features
            )
            
            if distill_loss.item() > 0:
                loss = loss + distill_loss
        
        # 构建输出
        output = {
            'logits': logits,  # ✓ 现在是head输出的logits
            'loss': loss
        }
        
        if return_expert_logits and expert_features_dict is not None:
            output['expert_features'] = expert_features_dict  # 返回专家特征
        
        return output
    
    def set_active_head(self, session_name: str, strict: bool = True):
        """设置当前活动的任务头"""
        self.current_session = session_name
        # ✓ 现在head会被真正使用！
        if self.head_manager.set_active_head(session_name, strict=strict):
            self.head = self.head_manager.get_current_head()
            return True
        return False
    
    def freeze_old_tasks(self):
        """冻结所有旧任务的专家"""
        for task_name in self.deqa_cl.task_ensembles.keys():
            if task_name != self.current_session:
                self.deqa_cl.freeze_task(task_name)
    
    def get_deqa_ensemble(self, task_name: str) -> Optional[DEQAEnsemble]:
        """获取指定任务的DEQA专家集成"""
        return self.deqa_cl.get_ensemble(task_name)

