"""
特征聚类可视化模块

用于观察持续学习过程中的特征表示变化和类别分布

新功能（v1.1.0）：
- 使用实际标签名（NEG/NEU/POS等）而不是Class 0/1/2
- 增强版可视化：同时显示真实标签和预测标签
"""

from .feature_clustering import (
    extract_features_and_labels,
    get_label_names_for_task,
    plot_tsne,
    plot_umap,
    plot_continual_learning_evolution,
    visualize_task_after_training,
    visualize_all_tasks_evolution
)

# 增强版功能
try:
    from .feature_clustering_enhanced import (
        extract_features_labels_and_predictions,
        plot_tsne_with_label_names,
        plot_tsne_comparison,
        visualize_task_enhanced
    )
    _enhanced_available = True
except ImportError:
    _enhanced_available = False

__all__ = [
    # 基础功能
    'extract_features_and_labels',
    'get_label_names_for_task',
    'plot_tsne',
    'plot_umap',
    'plot_continual_learning_evolution',
    'visualize_task_after_training',
    'visualize_all_tasks_evolution',
]

if _enhanced_available:
    __all__.extend([
        # 增强版功能
        'extract_features_labels_and_predictions',
        'plot_tsne_with_label_names',
        'plot_tsne_comparison',
        'visualize_task_enhanced'
    ])

__version__ = '1.1.0'
__author__ = 'MCL Project'

