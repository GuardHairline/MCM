# continual/deqa.py
"""
DEQA (Descriptions Enhanced Question-Answering Framework) for Continual Learning

基于论文: "DEQA: Descriptions Enhanced Question-Answering Framework 
for Multimodal Aspect-Based Sentiment Analysis"

核心思想:
1. 使用图像描述(GPT-4生成)作为额外的文本信息
2. 将任务框架为问答形式
3. 使用多专家集成决策:
   - Expert 1: Text-only (DeBERTa)
   - Expert 2: Text + Description (DeBERTa)
   - Expert 3: Text + CLIP Image Features
4. 通过投票/加权集成产生最终预测

持续学习适配:
- 为每个任务维护独立的专家集成
- 使用知识蒸馏保持旧专家的知识
- 支持冻结旧专家的参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy


class DEQAExpert(nn.Module):
    """
    单个DEQA专家模型
    
    Args:
        input_dim: 输入特征维度
        num_labels: 输出标签数（如果use_head=True，则不输出logits）
        expert_type: 专家类型 ('text', 'description', 'clip')
        hidden_dim: 隐藏层维度
        dropout_prob: Dropout概率
        use_head: 是否包含分类头（False时只输出特征）
    """
    def __init__(
        self,
        input_dim: int,
        num_labels: int = None,
        expert_type: str = 'text',
        hidden_dim: int = 768,
        dropout_prob: float = 0.1,
        use_head: bool = False  # ✓ 新增：控制是否输出logits
    ):
        super().__init__()
        self.expert_type = expert_type
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.use_head = use_head
        
        # 专家特定的特征转换层
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )
        
        # 分类头（可选）
        if use_head and num_labels is not None:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim // 2, num_labels)
            )
        else:
            self.classifier = None
        
    def forward(self, features: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            features: (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)
            return_features: 是否返回特征而不是logits
        Returns:
            如果return_features=True或不使用head: 返回特征
            否则: 返回logits
        """
        transformed = self.feature_transform(features)
        
        # ✓ 新增：支持返回特征
        if return_features or not self.use_head or self.classifier is None:
            return transformed
        
        logits = self.classifier(transformed)
        return logits


class DEQAEnsemble(nn.Module):
    """
    DEQA多专家集成模块
    
    Args:
        text_dim: 文本特征维度
        num_labels: 标签数量（如果use_head=True则可选）
        use_description: 是否使用描述专家
        use_clip: 是否使用CLIP专家
        ensemble_method: 集成方法 ('vote', 'weighted', 'learned')
        hidden_dim: 隐藏层维度
        dropout_prob: Dropout概率
        use_head: 是否在专家中包含分类头（False时输出融合特征）
    """
    def __init__(
        self,
        text_dim: int,
        num_labels: int = None,
        use_description: bool = True,
        use_clip: bool = True,
        ensemble_method: str = 'weighted',
        hidden_dim: int = 768,
        dropout_prob: float = 0.1,
        use_head: bool = False  # ✓ 新增：控制模式
    ):
        super().__init__()
        self.num_labels = num_labels
        self.ensemble_method = ensemble_method
        self.use_description = use_description
        self.use_clip = use_clip
        self.use_head = use_head
        self.hidden_dim = hidden_dim
        
        # 创建专家
        self.experts = nn.ModuleDict()
        
        # Expert 1: Text-only
        self.experts['text'] = DEQAExpert(
            input_dim=text_dim,
            num_labels=num_labels,
            expert_type='text',
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            use_head=use_head  # ✓ 传递use_head
        )
        
        # Expert 2: Text + Description
        if use_description:
            self.experts['description'] = DEQAExpert(
                input_dim=text_dim,  # 描述也用文本编码器编码
                num_labels=num_labels,
                expert_type='description',
                hidden_dim=hidden_dim,
                dropout_prob=dropout_prob,
                use_head=use_head  # ✓ 传递use_head
            )
        
        # Expert 3: Text + CLIP
        if use_clip:
            self.experts['clip'] = DEQAExpert(
                input_dim=text_dim,  # 融合后的特征维度
                num_labels=num_labels,
                expert_type='clip',
                hidden_dim=hidden_dim,
                dropout_prob=dropout_prob,
                use_head=use_head  # ✓ 传递use_head
            )
        
        # 集成权重
        num_experts = len(self.experts)
        if ensemble_method == 'learned':
            # 可学习的集成权重
            self.ensemble_weights = nn.Parameter(torch.ones(num_experts) / num_experts)
        elif ensemble_method == 'weighted':
            # 固定权重
            self.register_buffer('ensemble_weights', torch.ones(num_experts) / num_experts)
        # vote方法不需要权重
        
    def forward(
        self,
        text_features: torch.Tensor,
        description_features: Optional[torch.Tensor] = None,
        clip_features: Optional[torch.Tensor] = None,
        return_expert_logits: bool = False,
        return_features: bool = False  # ✓ 新增：是否返回融合特征
    ) -> torch.Tensor:
        """
        前向传播 - 多专家集成
        
        Args:
            text_features: 文本特征 (batch_size, seq_len, text_dim) 或 (batch_size, text_dim)
            description_features: 描述特征 (batch_size, seq_len, text_dim) 或 (batch_size, text_dim)
            clip_features: CLIP特征 (batch_size, seq_len, text_dim) 或 (batch_size, text_dim)
            return_expert_logits: 是否返回每个专家的logits/features
            return_features: 是否返回融合特征而不是logits
            
        Returns:
            如果return_features=True: 返回融合后的特征
            否则: 返回集成后的logits
            (可选) expert_outputs: 各专家的输出字典
        """
        expert_outputs = []
        expert_names = []
        
        # Expert 1: Text-only
        text_output = self.experts['text'](text_features, return_features=return_features)
        expert_outputs.append(text_output)
        expert_names.append('text')
        
        # Expert 2: Description
        if self.use_description and description_features is not None:
            desc_output = self.experts['description'](description_features, return_features=return_features)
            expert_outputs.append(desc_output)
            expert_names.append('description')
        
        # Expert 3: CLIP
        if self.use_clip and clip_features is not None:
            clip_output = self.experts['clip'](clip_features, return_features=return_features)
            expert_outputs.append(clip_output)
            expert_names.append('clip')
        
        # ✓ 新增：如果返回特征，直接加权融合
        if return_features or not self.use_head:
            ensemble_output = self._weighted_ensemble_features(expert_outputs)
        else:
            # 集成logits
            if self.ensemble_method == 'vote':
                # 硬投票
                ensemble_output = self._vote_ensemble(expert_outputs)
            elif self.ensemble_method in ['weighted', 'learned']:
                # 加权平均
                ensemble_output = self._weighted_ensemble(expert_outputs)
            else:
                raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        if return_expert_logits:
            expert_dict = {name: output for name, output in zip(expert_names, expert_outputs)}
            return ensemble_output, expert_dict
        
        return ensemble_output
    
    def _weighted_ensemble_features(self, expert_features: List[torch.Tensor]) -> torch.Tensor:
        """加权融合特征"""
        # 归一化权重
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # 加权求和
        ensemble_features = torch.zeros_like(expert_features[0])
        for i, feat in enumerate(expert_features):
            ensemble_features = ensemble_features + weights[i] * feat
        
        return ensemble_features
    
    def _vote_ensemble(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        """硬投票集成"""
        # 获取每个专家的预测
        predictions = [torch.argmax(logits, dim=-1) for logits in expert_outputs]
        
        # 投票
        # 将所有预测堆叠 (num_experts, batch_size, seq_len) 或 (num_experts, batch_size)
        stacked_preds = torch.stack(predictions, dim=0)
        
        # 对每个位置进行投票
        # 转换为one-hot然后求和
        batch_shape = stacked_preds.shape[1:]
        num_experts = len(expert_outputs)
        
        # 简单实现：使用mode (众数)
        # PyTorch没有直接的mode for last dim，我们用一个技巧
        votes = torch.zeros((*batch_shape, self.num_labels), device=stacked_preds.device)
        for i in range(num_experts):
            pred = stacked_preds[i]
            votes.scatter_add_(-1, pred.unsqueeze(-1), torch.ones_like(pred.unsqueeze(-1).float()))
        
        return votes  # 返回投票计数作为logits
    
    def _weighted_ensemble(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        """加权平均集成"""
        # 归一化权重
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # 加权求和
        ensemble_logits = torch.zeros_like(expert_outputs[0])
        for i, logits in enumerate(expert_outputs):
            ensemble_logits = ensemble_logits + weights[i] * logits
        
        return ensemble_logits
    
    def freeze_experts(self, expert_names: Optional[List[str]] = None):
        """冻结指定专家的参数"""
        if expert_names is None:
            expert_names = list(self.experts.keys())
        
        for name in expert_names:
            if name in self.experts:
                for param in self.experts[name].parameters():
                    param.requires_grad = False
    
    def unfreeze_experts(self, expert_names: Optional[List[str]] = None):
        """解冻指定专家的参数"""
        if expert_names is None:
            expert_names = list(self.experts.keys())
        
        for name in expert_names:
            if name in self.experts:
                for param in self.experts[name].parameters():
                    param.requires_grad = True


class DEQACL(nn.Module):
    """
    DEQA持续学习包装器
    
    为每个任务维护独立的专家集成，支持知识蒸馏和专家冻结
    
    Args:
        text_dim: 文本特征维度
        use_description: 是否使用描述专家
        use_clip: 是否使用CLIP专家
        ensemble_method: 集成方法
        distill_temperature: 蒸馏温度
        distill_weight: 蒸馏损失权重
        freeze_old_experts: 是否冻结旧任务的专家
    """
    def __init__(
        self,
        text_dim: int = 768,
        use_description: bool = True,
        use_clip: bool = True,
        ensemble_method: str = 'weighted',
        distill_temperature: float = 2.0,
        distill_weight: float = 0.5,
        freeze_old_experts: bool = True,
        hidden_dim: int = 768,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.text_dim = text_dim
        self.use_description = use_description
        self.use_clip = use_clip
        self.ensemble_method = ensemble_method
        self.distill_temperature = distill_temperature
        self.distill_weight = distill_weight
        self.freeze_old_experts = freeze_old_experts
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        
        # 存储每个任务的专家集成
        self.task_ensembles = nn.ModuleDict()
        
        # 存储旧任务的教师模型（用于蒸馏）
        self.teacher_ensembles = {}
        
        # 当前激活的任务
        self.current_task = None
    
    def add_task(self, task_name: str, num_labels: int = None, use_head: bool = False):
        """
        为新任务添加专家集成
        
        Args:
            task_name: 任务名称
            num_labels: 标签数量（如果use_head=True则需要）
            use_head: 是否在专家中包含分类头
        """
        # 创建新的专家集成
        ensemble = DEQAEnsemble(
            text_dim=self.text_dim,
            num_labels=num_labels,
            use_description=self.use_description,
            use_clip=self.use_clip,
            ensemble_method=self.ensemble_method,
            hidden_dim=self.hidden_dim,
            dropout_prob=self.dropout_prob,
            use_head=use_head  # ✓ 传递use_head参数
        )
        
        self.task_ensembles[task_name] = ensemble
        
        # 如果需要冻结旧专家
        if self.freeze_old_experts:
            for old_task in self.task_ensembles.keys():
                if old_task != task_name:
                    self.task_ensembles[old_task].freeze_experts()
        
        # 保存当前任务
        self.current_task = task_name
        
        # 为旧任务创建教师模型（用于蒸馏）
        if len(self.task_ensembles) > 1:
            for old_task in self.task_ensembles.keys():
                if old_task != task_name and old_task not in self.teacher_ensembles:
                    # 深拷贝并冻结
                    teacher = copy.deepcopy(self.task_ensembles[old_task])
                    teacher.eval()
                    for param in teacher.parameters():
                        param.requires_grad = False
                    self.teacher_ensembles[old_task] = teacher
    
    def forward(
        self,
        task_name: str,
        text_features: torch.Tensor,
        description_features: Optional[torch.Tensor] = None,
        clip_features: Optional[torch.Tensor] = None,
        return_expert_logits: bool = False,
        return_features: bool = False  # ✓ 新增
    ):
        """
        前向传播
        
        Args:
            task_name: 要使用的任务名称
            text_features: 文本特征
            description_features: 描述特征
            clip_features: CLIP特征
            return_expert_logits: 是否返回专家logits
            return_features: 是否返回融合特征而不是logits
            
        Returns:
            features/logits或(features/logits, expert_outputs)
        """
        if task_name not in self.task_ensembles:
            raise ValueError(f"Task {task_name} not found in DEQA ensembles")
        
        return self.task_ensembles[task_name](
            text_features=text_features,
            description_features=description_features,
            clip_features=clip_features,
            return_expert_logits=return_expert_logits,
            return_features=return_features  # ✓ 传递参数
        )
    
    def compute_distillation_loss(
        self,
        task_name: str,
        text_features: torch.Tensor,
        description_features: Optional[torch.Tensor] = None,
        clip_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算蒸馏损失
        
        Args:
            task_name: 当前任务名称
            text_features: 文本特征
            description_features: 描述特征
            clip_features: CLIP特征
            
        Returns:
            distill_loss: 蒸馏损失
        """
        if not self.teacher_ensembles:
            return torch.tensor(0.0, device=text_features.device)
        
        distill_loss = 0.0
        num_teachers = 0
        
        # 对每个旧任务计算蒸馏损失
        for old_task, teacher in self.teacher_ensembles.items():
            if old_task == task_name:
                continue
            
            with torch.no_grad():
                teacher_logits = teacher(
                    text_features=text_features,
                    description_features=description_features,
                    clip_features=clip_features
                )
            
            student_logits = self.task_ensembles[task_name](
                text_features=text_features,
                description_features=description_features,
                clip_features=clip_features
            )
            
            # KL散度蒸馏
            T = self.distill_temperature
            loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction='batchmean'
            ) * (T ** 2)
            
            distill_loss += loss
            num_teachers += 1
        
        if num_teachers > 0:
            distill_loss = distill_loss / num_teachers
        
        return distill_loss * self.distill_weight
    
    def get_ensemble(self, task_name: str) -> DEQAEnsemble:
        """获取指定任务的专家集成"""
        return self.task_ensembles.get(task_name, None)
    
    def freeze_task(self, task_name: str):
        """冻结指定任务的所有专家"""
        if task_name in self.task_ensembles:
            self.task_ensembles[task_name].freeze_experts()
    
    def unfreeze_task(self, task_name: str):
        """解冻指定任务的所有专家"""
        if task_name in self.task_ensembles:
            self.task_ensembles[task_name].unfreeze_experts()

