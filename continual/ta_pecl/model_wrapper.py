# continual/ta_pecl/model_wrapper.py
import torch
import torch.nn as nn
import json
import os
from .config import get_expert_config, TASK_NAME_MAP 
from .modules import TA_PECL_Block

class TaskState:
    """
    一个简单的非 nn.Module 类，用于在父子模块间共享状态（task_id）。
    因为不是 nn.Module，所以不会触发 PyTorch 的递归遍历死循环。
    """
    def __init__(self):
        self.current_task_id = 0

class TA_PECL_LayerWrapper(nn.Module):
    """
    单层包装器：拦截原始 Transformer 层的输出，并注入 Adapter 信号。
    """
    def __init__(self, original_layer, adapter, task_state):
        super().__init__()
        self.original_layer = original_layer
        self.adapter = adapter
        self.task_state = task_state  # 引用共享状态对象

    def forward(self, *args, **kwargs):
        # 1. 执行原始层 (冻结状态)
        # 注意：这里我们让原始层自己处理 attention_mask, relative_pos 等复杂参数
        with torch.no_grad():
            outputs = self.original_layer(*args, **kwargs)
        
        # DeBERTa/BERT 的输出通常是 tuple: (hidden_states, attention_weights, ...)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
            
        # 2. 执行 Adapter (可训练)
        # 从共享状态获取当前任务 ID
        task_id = self.task_state.current_task_id
        
        adapter_out = self.adapter(hidden_states, task_id)
        
        # 3. 残差连接 (Residual Connection)
        hidden_states = hidden_states + adapter_out
        
        # 4. 恢复原始输出格式
        if isinstance(outputs, tuple):
            return (hidden_states,) + outputs[1:]
        else:
            return hidden_states
    def reset_expert_stats(self):
        """重置所有层的专家统计"""
        for layer in self.patched_layers:
            # layer 是 TA_PECL_LayerWrapper
            # layer.adapter 是 TA_PECL_Block
            # layer.adapter.router 是 TaskAwareRouter
            layer.adapter.router.reset_stats()
        print("[TA-PECL] Expert statistics reset.")

    
class TA_PECL_ModelWrapper(nn.Module):
    """
    TA-PECL 模型包装器：
    不重写 forward，而是通过 '手术' 替换 base_model 内部的 Transformer 层。
    这种方法最稳健，兼容 DeBERTa, BERT, RoBERTa 等多种架构。
    """
    def __init__(self, base_model, args):
        super().__init__()
        self.base_model = base_model
        self.args = args
        
        # 配置
        self.hidden_size = getattr(args, 'hidden_dim', 768)
        self.expert_config = get_expert_config(hidden_size=self.hidden_size)
        self.num_tasks = len(TASK_NAME_MAP)
        self.top_k = getattr(args, 'ta_pecl_top_k', 4)
        
        # [关键修复] 使用独立的状态对象，避免 nn.Module 循环引用
        self.task_state = TaskState()
        
        output_dir = os.path.dirname(args.output_model_path) if hasattr(args, 'output_model_path') and args.output_model_path else "./checkpoints"
        self.stats_dir = os.path.join(output_dir, "expert_stats")
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # 1. 自动定位并替换 Transformer 层
        self.patched_layers = self._find_and_replace_layers()
        
        # 2. 冻结主干 (除了我们刚注入的 Adapter)
        self._freeze_backbone()
        
        print(f"\n[TA-PECL] System Initialized Successfully.")
        print(f"          - Strategy: Layer Injection (In-place)")
        print(f"          - Injected Layers: {len(self.patched_layers)}")
        print(f"          - Active Experts: Top-{self.top_k}")

    def _find_and_replace_layers(self):
        """
        递归查找 transformer layers 并进行替换
        """
        # 策略 1: 针对你的 BaseMultimodalModel (DeBERTa V3)
        if hasattr(self.base_model, 'text_encoder'):
            # DeBERTa V3 结构: text_encoder -> encoder -> layer (ModuleList)
            encoder_module = self.base_model.text_encoder
            if hasattr(encoder_module, 'encoder'):
                 container = encoder_module.encoder
                 if hasattr(container, 'layer'):
                     return self._replace_in_container(container, 'layer')
                 elif hasattr(container, 'layers'):
                     return self._replace_in_container(container, 'layers')
        
        # 策略 2: 标准 HF Model (base_model 本身就是 Transformer)
        if hasattr(self.base_model, 'encoder'):
            container = self.base_model.encoder
            if hasattr(container, 'layer'):
                return self._replace_in_container(container, 'layer')
        
        raise ValueError("TA-PECL Error: Could not locate transformer layers to patch in base_model.")

    def _replace_in_container(self, container, attribute_name):
        """
        在 ModuleList 容器中执行原地替换
        """
        layers_list = getattr(container, attribute_name) # 获取 ModuleList 对象
        patched_layers = []
        
        for i, original_layer in enumerate(layers_list):
            # 防止重复包装
            if isinstance(original_layer, TA_PECL_LayerWrapper):
                patched_layers.append(original_layer)
                continue

            # 创建 Adapter Block
            adapter_block = TA_PECL_Block(
                hidden_size=self.hidden_size, 
                num_tasks=self.num_tasks,
                expert_config=self.expert_config,
                top_k=self.top_k
            )
            
            # 创建包装层 (LayerWrapper)
            # [关键修复] 传入 task_state 而不是 self
            wrapped_layer = TA_PECL_LayerWrapper(original_layer, adapter_block, self.task_state)
            
            # [关键] 原地替换！
            layers_list[i] = wrapped_layer
            patched_layers.append(wrapped_layer)
            
        return patched_layers

    def _freeze_backbone(self):
        # 1. 先冻结所有参数
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 2. 解冻 Adapter 参数
        # 因为 Adapter 现在是 base_model 的一部分，我们需要通过 patched_layers 找到它们
        count = 0
        for layer in self.patched_layers:
            for param in layer.adapter.parameters():
                param.requires_grad = True
                count += 1
                
        print(f"[TA-PECL] Backbone frozen. {count} Adapter parameter groups unfrozen.")

    def set_task_name(self, task_name):
        """设置当前任务ID，LayerWrapper 会通过 task_state 读取它"""
        t_name = task_name.lower()
        found = False
        for key, tid in TASK_NAME_MAP.items():
            if key in t_name:
                self.task_state.current_task_id = tid # 更新共享状态
                found = True
                break
        if not found:
            print(f"[Warning] Unknown task name '{task_name}', defaulting to MASC (id=0).")
            self.task_state.current_task_id = 0

    def forward(self, *args, **kwargs):
        """
        直接委托给 base_model。
        由于我们已经替换了内部的层，base_model 的 forward 流程会自动经过我们的 Adapter。
        """
        # 确保 forward 前状态已设置（虽然通常由 set_task_name 处理）
        # 这里只做委托
        return self.base_model(*args, **kwargs)
    # 重置统计，确保不同任务不混淆
    def reset_expert_stats(self):
        """重置所有层的专家统计数据"""
        for layer in self.patched_layers:
            # layer.adapter.router 是 TaskAwareRouter
            if hasattr(layer.adapter.router, 'reset_stats'):
                layer.adapter.router.reset_stats()
        # print("[TA-PECL] Expert statistics reset.") # 可选：减少日志刷屏

    # 核心保存逻辑
    def save_expert_stats(self, session_name, phase="train", epoch=None):
        """
        将专家统计信息保存为 JSON 文件
        Args:
            session_name: 当前任务会话名称 (如 masc_twitter2015)
            phase: 阶段 (train, eval, test)
            epoch: 当前轮数 (可选)
        """
        # 1. 汇总所有层的数据
        total_samples = 0
        global_counts = None
        global_weights = None
        
        # 遍历所有层累加
        for layer in self.patched_layers:
            router = layer.adapter.router
            if global_counts is None:
                global_counts = router.activation_counts.clone().cpu()
                global_weights = router.accumulated_weights.clone().cpu()
                total_samples = router.total_samples
            else:
                global_counts += router.activation_counts.cpu()
                global_weights += router.accumulated_weights.cpu()
        
        # 转换为 Python 数字
        if torch.is_tensor(total_samples):
            total_samples = total_samples.item()

        if total_samples == 0:
            return

        # 2. 构建统计字典
        stats_data = {
            "session_name": session_name,
            "phase": phase,
            "epoch": epoch,
            "total_samples": total_samples,
            "top_k": self.top_k,
            "num_layers": len(self.patched_layers),
            "experts": {}
        }

        # 计算总决策次数 (用于算百分比)
        total_decisions = total_samples * len(self.patched_layers) * self.top_k
        expert_names = list(self.expert_config.keys())

        for idx, name in enumerate(expert_names):
            count = int(global_counts[idx].item())
            weight_sum = global_weights[idx].item()
            
            stats_data["experts"][name] = {
                "activation_count": count,            # 激活总次数
                "accumulated_weight": weight_sum,     # 权重总和
                "active_rate": (count / total_decisions), # 激活占比 (0~1)
                "avg_weight": (weight_sum / count) if count > 0 else 0.0 # 被选中时的平均权重
            }

        # 3. 写入文件
        # 文件名示例: stats_masc_twitter2015_train_final.json
        filename = f"stats_{session_name}_{phase}"
        if epoch is not None:
            filename += f"_ep{epoch}"
        filename += ".json"
        
        save_path = os.path.join(self.stats_dir, filename)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=4)
            print(f"[TA-PECL] 📊 Expert stats saved to: {save_path}")
        except Exception as e:
            print(f"[TA-PECL] Failed to save stats: {e}")

    # 日志打印方法，方便在控制台快速查看
    def log_expert_statistics(self, logger, phase="TRAIN"):
        """
        汇总并打印专家使用情况报告
        """
        total_samples = 0
        
        # 聚合所有层的统计数据
        # global_counts: [num_experts]
        global_counts = None
        global_weights = None
        
        # 1. 汇总数据
        for layer in self.patched_layers:
            router = layer.adapter.router
            if global_counts is None:
                global_counts = router.activation_counts.clone()
                global_weights = router.accumulated_weights.clone()
                total_samples = router.total_samples
            else:
                global_counts += router.activation_counts
                global_weights += router.accumulated_weights
                # total_samples 在所有层应该是一样的，取一个即可
        
        if total_samples == 0:
            logger.warning("[TA-PECL] No statistics collected (total_samples=0).")
            return

        # 2. 计算百分比
        # 激活率 = 激活次数 / (总样本数 * 层数 * TopK) ? 
        # 更直观的是：每个样本平均激活该专家的层数比例，或者简单的总占比
        # 这里我们计算：在所有路由决策中（层数*样本数*TopK），该专家被选中的概率
        
        num_layers = len(self.patched_layers)
        total_decisions = total_samples * num_layers * self.top_k
        
        # 3. 打印报告
        logger.info("=" * 80)
        logger.info(f"📊 TA-PECL Expert Usage Report ({phase}) - Total Samples: {total_samples}")
        logger.info(f"{'Expert Name':<20} | {'Type':<10} | {'Active %':<10} | {'Avg Weight':<10} | {'Count':<10}")
        logger.info("-" * 80)
        
        expert_names = list(self.expert_config.keys())
        
        # 按类型分组排序以便查看 (Task -> Modality -> DEQA -> Flex)
        sorted_indices = sorted(range(len(expert_names)), key=lambda k: expert_names[k])
        
        for idx in sorted_indices:
            name = expert_names[idx]
            count = int(global_counts[idx].item())
            weight_sum = global_weights[idx].item()
            
            # Active %: 该专家被激活的频率
            active_pct = (count / total_decisions) * 100
            
            # Avg Weight: 被激活时的平均权重 (避免除以0)
            avg_weight = (weight_sum / count) if count > 0 else 0.0
            
            # 确定类型
            etype = "Unknown"
            if "flex" in name: etype = "Flexible"
            elif "deqa" in name: etype = "DEQA"
            elif "text" in name or "multi" in name: etype = "Modal"
            else: etype = "Task"
            
            # 高亮显示过度活跃的 Flex 专家 (例如超过 20%)
            highlight = ""
            if etype == "Flexible" and active_pct > 20:
                highlight = "🔴 (High)"
            elif etype == "Task" and active_pct < 1:
                highlight = "⚠️ (Low)"

            logger.info(f"{name:<20} | {etype:<10} | {active_pct:6.2f}%    | {avg_weight:6.4f}     | {count:<10} {highlight}")
            
        logger.info("=" * 80)