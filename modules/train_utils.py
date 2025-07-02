# modules/train_utils.py
import os
import json
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple

from models.base_model import BaseMultimodalModel
from continual.ewc import MultiTaskEWC
from continual.lwf import LwFDistiller
from continual.si import SynapticIntelligence
from continual.mas import MASRegularizer
from continual.pnn import PNNManager
from continual.gem import GEMManager
from continual.tam_cl import TamCLModel
from continual.moe_adapters.moe_model_wrapper import MoeAdapterWrapper
from continual.moe_adapters.ddas_router import DDASRouter
from continual.clap4clip.clap4clip import CLAP4CLIP
from continual.label_embedding_manager import LabelEmbeddingManager
from continual.metrics import ContinualMetrics
from continual.experience_replay import ExperienceReplayMemory, make_dynamic_replay_condition
from datasets.get_dataset import get_dataset


class Full_Model(nn.Module):
    def __init__(self, base_model, head, dropout_prob=0.1):
        super().__init__()
        self.base_model = base_model
        self.head = head
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, attention_mask, token_type_ids, image_tensor):
        fused_feat = self.base_model(input_ids, attention_mask, token_type_ids, image_tensor)
        fused_feat = self.dropout(fused_feat)
        logits = self.head(fused_feat)
        return logits


def load_train_info(train_info_path: str) -> Dict[str, Any]:
    """加载训练信息"""
    train_info = {}
    if os.path.exists(train_info_path):
        with open(train_info_path, "r", encoding="utf-8") as f:
            try:
                train_info = json.load(f)
            except:
                train_info = {}
    
    # 初始化默认结构
    if "tasks" not in train_info:
        train_info["tasks"] = []
    if "acc_matrix" not in train_info:
        train_info["acc_matrix"] = []
    if "sessions" not in train_info:
        train_info["sessions"] = []
    
    return train_info


def create_model(args, device: str, label_embedding_manager: Optional[LabelEmbeddingManager] = None):
    """创建模型"""
    if args.use_label_embedding:
        from models.task_heads.get_head_new import get_head
    else:
        from models.task_heads.get_head import get_head
    if args.tam_cl:
        model = TamCLModel(
            text_model_name=args.text_model_name,
            image_model_name=args.image_model_name,
            fusion_strategy=args.fusion_strategy,
            num_heads=args.num_heads,
            mode=args.mode,
            hidden_dim=args.hidden_dim,
            dropout_prob=args.dropout_prob
        )
        if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
            pretrained = torch.load(args.pretrained_model_path)
            model.base_model.load_state_dict(pretrained, strict=False)
        model.add_task(args.session_name, task_name=args.task_name, num_labels=args.num_labels, args=args)
        return model.to(device)
    
    elif args.moe_adapters:
        base_model = BaseMultimodalModel(
            args.text_model_name, args.image_model_name,
            multimodal_fusion=args.fusion_strategy,
            num_heads=args.num_heads, mode=args.mode
        )
        moe_model = MoeAdapterWrapper(base_model,
                                      num_experts=args.moe_num_experts,
                                      top_k=args.moe_top_k)
        moe_model.start_new_task()
        
        # 使用标签嵌入
        label_emb = label_embedding_manager.get_embedding() if label_embedding_manager else None
        
        head = get_head(args.task_name, moe_model.base_model, args, label_emb=label_emb)
        
        full_model = Full_Model(moe_model, head, dropout_prob=args.dropout_prob).to(device)
        
        if args.ddas:
            full_model.ddas = DDASRouter(
                feature_dim=base_model.text_hidden_size,
                threshold=args.ddas_threshold
            ).to(device)
        else:
            full_model.ddas = None
        
        return full_model
    
    elif args.clap4clip:
        return CLAP4CLIP(
            text_model_name=args.text_model_name,
            image_model_name=args.image_model_name,
            num_labels=args.num_labels,
            dropout_prob=args.dropout_prob
        ).to(device)
    
    else:
        base_model = BaseMultimodalModel(
            args.text_model_name,
            args.image_model_name,
            multimodal_fusion=args.fusion_strategy,
            num_heads=args.num_heads,
            mode=args.mode
        )
        if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
            base_model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)

        # 使用标签嵌入
        label_emb = label_embedding_manager.get_embedding() if label_embedding_manager else None
        head = get_head(args.task_name, base_model, args, label_emb=label_emb)
        
        return Full_Model(base_model, head, dropout_prob=args.dropout_prob).to(device)


def create_continual_learning_components(args, full_model, train_info: Dict[str, Any], device: str, logger=None):
    """创建持续学习组件"""
    ewc = None
    fisher_selector = None
    replay_memory = None
    lwf = None
    si = None
    mas = None
    gem = None
    pnn = None
    
    old_sessions_count = len(train_info.get("sessions", []))
    
    # EWC 或 MyMethod
    if args.ewc == 1 or args.mymethod == 1:
        shared_ewc = MultiTaskEWC(
            model=full_model,
            current_task_name=args.task_name,
            session_name=args.session_name,
            num_labels=args.num_labels,
            ewc_lambda=args.ewc_lambda,
            ewc_dir=args.ewc_dir
        )
        
        if args.mymethod == 1:
            fisher_selector = shared_ewc
            if logger:
                logger.info("[MyMethod]")
        if args.ewc == 1:
            ewc = shared_ewc
            if logger:
                logger.info("[EWC]")
            
        if old_sessions_count > 0:
            if args.ewc == 1:
                ewc.load_all_previous_tasks(train_info)
            else:
                fisher_selector.load_all_previous_tasks(train_info)
    
    # Experience Replay
    if args.replay == 1:
        replay_memory = ExperienceReplayMemory()
        dynamic_condition = make_dynamic_replay_condition(
            train_info.get("sessions", []), 
            threshold_factor=0.9
        )
        
        for hist_session in train_info.get("sessions", []):
            replay_memory.add_session_memory_buffer(
                session_info=hist_session,
                memory_percentage=args.memory_percentage,
                replay_ratio=args.replay_ratio,
                replay_frequency=args.replay_frequency,
                replay_condition=dynamic_condition
            )
        if logger:
            logger.info(f"[Replay] percentage={args.memory_percentage}, ratio={args.replay_ratio}, frequency={args.replay_frequency}")
    
    # LwF
    if args.lwf:
        alpha_t = args.lwf_alpha / (1 + args.lwf_decay * old_sessions_count)
        old_model = copy.deepcopy(full_model)
        old_model.eval()
        lwf = LwFDistiller(old_model, T=args.lwf_T, alpha=alpha_t)
        if logger:
            logger.info(f"[LwF] base_alpha={args.lwf_alpha}, decay={args.lwf_decay}, alpha_t={alpha_t:.4f}")
    
    # SI
    if args.si:
        eps_t = args.si_epsilon / (1 + args.si_decay * old_sessions_count)
        si = SynapticIntelligence(full_model, epsilon=eps_t)
        if logger:
            logger.info(f"[SI] base_eps={args.si_epsilon}, decay={args.si_decay}, eps_t={eps_t:.6f}")
    
    # MAS
    if args.mas:
        mas = MASRegularizer(full_model, epsilon=args.mas_eps)
        if logger:
            logger.info(f"[MAS] base_eps={args.mas_eps}")
    
    # GEM
    if args.gem:
        if logger:
            logger.info(f"[GEM] memory_size={args.gem_mem}")
        gem = GEMManager(full_model, memory_size=args.gem_mem, mem_dir=args.gem_mem_dir)
        for session in train_info["sessions"]:
            sess_args = argparse.Namespace(**session["args"])
            old_ds = get_dataset(session["task_name"], "train", sess_args)
            gem.register_task(session["task_name"], old_ds)
    
    # PNN
    if args.pnn:
        if logger:
            logger.info("[PNN]")
        pnn = PNNManager(
            args.text_model_name, args.image_model_name,
            args.fusion_strategy, args.num_heads, args.mode, args.hidden_dim
        )
        # 新 column 取代 full_model
        full_model = pnn.add_task(args.num_labels).to(device)
    
    return ewc, fisher_selector, replay_memory, lwf, si, mas, gem, pnn


def create_session_info(args) -> Dict[str, Any]:
    """创建会话信息"""
    return {
        "session_name": args.session_name,
        "task_name": args.task_name,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": args.epochs,
        "details": {},
        "final_metrics": None,
        "args": vars(args),
    }


def save_train_info(train_info: Dict[str, Any], train_info_path: str):
    """保存训练信息"""
    with open(train_info_path, "w", encoding="utf-8") as f:
        json.dump(train_info, f, indent=2, ensure_ascii=False)


def get_class_weights(task_name: str, device: str) -> Optional[torch.Tensor]:
    """获取类别权重"""
    if task_name == "mate":
        return torch.tensor([1.0, 15.0, 15.0], device=device)
    elif task_name == "mner":
        return torch.tensor([0.1, 164.0, 10.0, 270.0, 27.0, 340.0, 16.0, 360.0, 2.0], device=device)
    elif task_name == "mabsa":
        return torch.tensor([1.0, 3700.0, 234.0, 480.0, 34.0, 786.0, 69.0], device=device)
    else:
        return None


def is_sequence_task(task_name: str) -> bool:
    """判断是否为序列标注任务"""
    return task_name in ["mate", "mner", "mabsa"]


def create_optimizer(model, args):
    """创建优化器"""
    if args.moe_adapters:
        optim_params = [p for p in model.parameters() if p.requires_grad]
    else:
        optim_params = model.parameters()
    
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    return optimizer, scheduler


def create_ddas_optimizer(model, args):
    """创建DDAS优化器"""
    if args.ddas and hasattr(model, 'ddas') and model.ddas is not None:
        return torch.optim.Adam(model.ddas.parameters(), lr=1e-4)
    return None 