# continual/label_config.py
"""
统一的标签配置和映射管理
解决问题：标签映射分散、不一致、硬编码等问题
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """任务类型枚举"""
    TOKEN_LEVEL = "token"  # 序列标注任务
    SENTENCE_LEVEL = "sentence"  # 句子分类任务


@dataclass
class TaskLabelConfig:
    """单个任务的标签配置"""
    task_name: str
    task_type: TaskType
    num_labels: int
    label_names: List[str]
    label_descriptions: Dict[int, str]  # 标签ID到描述的映射
    sentiment_mapping: Optional[Dict[int, Tuple[int, int]]] = None  # 情感值到BIO标签的映射（用于MABSA）
    
    def get_label_text(self, label_id: int) -> str:
        """获取标签的文本描述"""
        return self.label_descriptions.get(label_id, f"label_{label_id}")
    
    def validate_label_id(self, label_id: int) -> bool:
        """验证标签ID是否有效"""
        return 0 <= label_id < self.num_labels


class UnifiedLabelManager:
    """统一的标签管理器"""
    
    def __init__(self):
        self._task_configs: Dict[str, TaskLabelConfig] = {}
        self._global_label_mapping: Dict[Tuple[str, int], int] = {}
        self._reverse_mapping: Dict[int, Tuple[str, int]] = {}
        self._initialize_task_configs()
        self._build_global_mapping()
    
    def _initialize_task_configs(self):
        """初始化所有任务的标签配置"""
        
        # MABSA: 多模态端到端情感分析
        self._task_configs["mabsa"] = TaskLabelConfig(
            task_name="mabsa",
            task_type=TaskType.TOKEN_LEVEL,
            num_labels=7,
            label_names=["O", "B-NEG", "I-NEG", "B-NEU", "I-NEU", "B-POS", "I-POS"],
            label_descriptions={
                0: "outside",
                1: "begin negative aspect",
                2: "inside negative aspect",
                3: "begin neutral aspect",
                4: "inside neutral aspect",
                5: "begin positive aspect",
                6: "inside positive aspect"
            },
            sentiment_mapping={
                -1: (1, 2),  # negative -> (B-NEG, I-NEG)
                0: (3, 4),   # neutral -> (B-NEU, I-NEU)
                1: (5, 6)    # positive -> (B-POS, I-POS)
            }
        )
        
        # MASC: 多模态方面情感分类
        self._task_configs["masc"] = TaskLabelConfig(
            task_name="masc",
            task_type=TaskType.SENTENCE_LEVEL,
            num_labels=3,
            label_names=["NEG", "NEU", "POS"],
            label_descriptions={
                0: "negative sentiment",
                1: "neutral sentiment",
                2: "positive sentiment"
            },
            sentiment_mapping={
                -1: (0, 0),  # negative -> label 0
                0: (1, 1),   # neutral -> label 1
                1: (2, 2)    # positive -> label 2
            }
        )
        
        # MATE: 多模态方面术语提取
        self._task_configs["mate"] = TaskLabelConfig(
            task_name="mate",
            task_type=TaskType.TOKEN_LEVEL,
            num_labels=3,
            label_names=["O", "B", "I"],
            label_descriptions={
                0: "outside aspect term",
                1: "begin aspect term",
                2: "inside aspect term"
            }
        )
        
        # MNER: 多模态命名实体识别
        self._task_configs["mner"] = TaskLabelConfig(
            task_name="mner",
            task_type=TaskType.TOKEN_LEVEL,
            num_labels=9,
            label_names=[
                "O", 
                "B-PER", "I-PER",
                "B-ORG", "I-ORG",
                "B-LOC", "I-LOC",
                "B-MISC", "I-MISC"
            ],
            label_descriptions={
                0: "outside entity",
                1: "begin person entity",
                2: "inside person entity",
                3: "begin organization entity",
                4: "inside organization entity",
                5: "begin location entity",
                6: "inside location entity",
                7: "begin miscellaneous entity",
                8: "inside miscellaneous entity"
            }
        )
    
    def _build_global_mapping(self):
        """构建全局标签映射"""
        global_idx = 0
        for task_name, config in self._task_configs.items():
            for label_id in range(config.num_labels):
                key = (task_name, label_id)
                self._global_label_mapping[key] = global_idx
                self._reverse_mapping[global_idx] = key
                global_idx += 1
    
    def get_task_config(self, task_name: str) -> Optional[TaskLabelConfig]:
        """获取任务的标签配置"""
        return self._task_configs.get(task_name)
    
    def get_global_label_id(self, task_name: str, local_label_id: int) -> Optional[int]:
        """将任务局部标签ID转换为全局标签ID"""
        return self._global_label_mapping.get((task_name, local_label_id))
    
    def get_task_and_label(self, global_label_id: int) -> Optional[Tuple[str, int]]:
        """将全局标签ID转换回任务名和局部标签ID"""
        return self._reverse_mapping.get(global_label_id)
    
    def get_label_text_mapping(self) -> Dict[Tuple[str, int], str]:
        """获取所有标签的文本描述映射（用于label embedding初始化）"""
        mapping = {}
        for task_name, config in self._task_configs.items():
            for label_id in range(config.num_labels):
                key = (task_name, label_id)
                mapping[key] = config.get_label_text(label_id)
        return mapping
    
    def get_label2idx(self) -> Dict[Tuple[str, int], int]:
        """获取全局标签映射（兼容现有代码）"""
        return self._global_label_mapping.copy()
    
    def get_task_num_labels(self, task_name: str) -> int:
        """获取任务的标签数量"""
        config = self._task_configs.get(task_name)
        return config.num_labels if config else 0
    
    def is_token_level_task(self, task_name: str) -> bool:
        """判断是否为token级任务"""
        config = self._task_configs.get(task_name)
        return config.task_type == TaskType.TOKEN_LEVEL if config else False
    
    def is_sentence_level_task(self, task_name: str) -> bool:
        """判断是否为句子级任务"""
        config = self._task_configs.get(task_name)
        return config.task_type == TaskType.SENTENCE_LEVEL if config else False
    
    def get_sentiment_labels(self, task_name: str, sentiment: int) -> Optional[Tuple[int, int]]:
        """
        根据情感值获取对应的标签
        
        Args:
            task_name: 任务名称
            sentiment: 情感值（-1, 0, 1）
            
        Returns:
            (B标签, I标签) 或 (标签, 标签) for sentence-level
        """
        config = self._task_configs.get(task_name)
        if config and config.sentiment_mapping:
            return config.sentiment_mapping.get(sentiment)
        return None
    
    def get_class_weights(self, task_name: str, device: str = "cpu") -> Optional['torch.Tensor']:
        """
        获取任务的类别权重（用于处理类别不平衡）
        
        注意：这些权重基于数据集统计，应该定期更新
        """
        import torch
        
        # 基于Twitter2015真实数据分布的统计 (样本数：MASC=250, MNER=50, MATE=250, MABSA=250)
        # 分析工具: python -m tools.analyze_label_distribution --task all --dataset twitter2015
        weights = {
            # MASC (句级情感): NEG=10.8%, NEU=59.6%, POS=29.6%
            # 策略: NEU占多数(60%)，大幅提升NEG权重(最少)，适度提升POS
            "masc": [5.0, 1.0, 2.0],  # [NEG, NEU, POS]
            
            # MNER (命名实体): O=95.25%, 实体类型共4.75%
            # 策略: O权重必须极低(0.1)，按实体稀有度分配权重
            # 格式: [O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC]
            "mner": [0.1, 10.0, 10.0, 18.0, 13.0, 12.0, 13.0, 13.0, 16.0],
            
            # MATE (Token级情感): O=90.06%, B=5.15%, I=4.80%
            # 策略: O权重降至0.5，B/I给5倍权重
            "mate": [0.5, 5.0, 5.0],  # [O, B, I]
            
            # MABSA (端到端情感): O=90.06%, NEG极少(0.5%), NEU相对多(3%), POS中等(1.5%)
            # 策略: O权重0.5，NEG最高(20x)，NEU适度(6x)，POS较高(10x)
            # 格式: [O, B-NEG, I-NEG, B-NEU, I-NEU, B-POS, I-POS]
            "mabsa": [0.5, 20.0, 20.0, 6.0, 6.0, 10.0, 10.0],
        }
        
        weight_values = weights.get(task_name)
        if weight_values is None:
            return None
        
        return torch.tensor(weight_values, dtype=torch.float32, device=device)
    
    def create_label_groups(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        创建语义相关的标签分组（用于label embedding相似度正则化）
        """
        return {
            "O": [  # 非目标标签
                ("mabsa", 0), ("mate", 0), ("mner", 0)
            ],
            "NEG": [  # 负向情感/否定
                ("mabsa", 1), ("mabsa", 2),  # B-NEG, I-NEG
                ("masc", 0)   # NEG
            ],
            "NEU": [  # 中性情感
                ("mabsa", 3), ("mabsa", 4),  # B-NEU, I-NEU
                ("masc", 1)   # NEU
            ],
            "POS": [  # 正向情感/肯定
                ("mabsa", 5), ("mabsa", 6),  # B-POS, I-POS
                ("masc", 2)   # POS
            ],
            "B_ENTITY": [  # 实体开始标签
                ("mate", 1),      # B
                ("mner", 1), ("mner", 3), ("mner", 5), ("mner", 7),  # B-PER, B-ORG, B-LOC, B-MISC
            ],
            "I_ENTITY": [  # 实体内部标签
                ("mate", 2),      # I
                ("mner", 2), ("mner", 4), ("mner", 6), ("mner", 8),  # I-PER, I-ORG, I-LOC, I-MISC
            ]
        }
    
    def validate_labels(self, task_name: str, labels: List[int]) -> Tuple[bool, str]:
        """
        验证标签列表是否有效
        
        Returns:
            (is_valid, error_message)
        """
        config = self._task_configs.get(task_name)
        if config is None:
            return False, f"Unknown task: {task_name}"
        
        for label_id in labels:
            if label_id == -100:  # padding label
                continue
            if not config.validate_label_id(label_id):
                return False, f"Invalid label_id {label_id} for task {task_name} (valid range: 0-{config.num_labels-1})"
        
        return True, ""
    
    def print_summary(self):
        """打印标签配置摘要"""
        print("=" * 80)
        print("Unified Label Manager Summary")
        print("=" * 80)
        
        for task_name, config in self._task_configs.items():
            print(f"\n[{task_name.upper()}] {config.task_type.value}-level task")
            print(f"  Number of labels: {config.num_labels}")
            print(f"  Label names: {', '.join(config.label_names)}")
            
            # 打印全局映射
            print(f"  Global mapping:")
            for label_id in range(config.num_labels):
                global_id = self._global_label_mapping[(task_name, label_id)]
                desc = config.get_label_text(label_id)
                print(f"    {label_id} -> {global_id} ({desc})")
        
        print(f"\n{'=' * 80}")
        print(f"Total global labels: {len(self._global_label_mapping)}")
        print("=" * 80)


# 全局单例
_global_label_manager = None

def get_label_manager() -> UnifiedLabelManager:
    """获取全局标签管理器单例"""
    global _global_label_manager
    if _global_label_manager is None:
        _global_label_manager = UnifiedLabelManager()
    return _global_label_manager


# 兼容性函数（用于替换现有代码）
def build_global_label_mapping() -> Dict[Tuple[str, int], int]:
    """兼容函数：构建全局标签映射"""
    return get_label_manager().get_label2idx()


def get_label_text_mapping() -> Dict[Tuple[str, int], str]:
    """兼容函数：获取标签文本映射"""
    return get_label_manager().get_label_text_mapping()


def create_label_groups() -> Dict[str, List[Tuple[str, int]]]:
    """兼容函数：创建标签分组"""
    return get_label_manager().create_label_groups()


def is_sequence_task(task_name: str) -> bool:
    """兼容函数：判断是否为序列任务"""
    return get_label_manager().is_token_level_task(task_name)


def get_class_weights(task_name: str, device: str = "cpu"):
    """兼容函数：获取类别权重"""
    return get_label_manager().get_class_weights(task_name, device)

