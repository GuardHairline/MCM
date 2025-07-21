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
import argparse


class Full_Model(nn.Module):
    def __init__(self, base_model, head, dropout_prob=0.1):
        super().__init__()
        self.base_model = base_model
        self.head = head
        self.dropout = nn.Dropout(dropout_prob)
        # 存储所有任务的模型头
        self.task_heads = {}
        self.current_session = None

    def forward(self, input_ids, attention_mask, token_type_ids, image_tensor):
        # 检查当前任务是否为序列标注任务
        current_task = self.get_current_task_name()
        is_seq_task = current_task in ["mate", "mner", "mabsa"]
        
        # 根据任务类型决定是否返回序列特征
        fused_feat = self.base_model(input_ids, attention_mask, token_type_ids, image_tensor, return_sequence=is_seq_task)
        fused_feat = self.dropout(fused_feat)
        logits = self.head(fused_feat)
        return logits
    
    def add_task_head(self, session_name: str, task_name: str, head, args):
        """添加任务特定的模型头"""
        self.task_heads[session_name] = {
            'head': head,
            'task_name': task_name,
            'args': args
        }
    
    def set_active_head(self, session_name: str):
        """设置活动模型头"""
        if session_name in self.task_heads:
            new_head = self.task_heads[session_name]['head']
            # 确保新的模型头在正确的设备上
            if hasattr(self, 'base_model'):
                device = next(self.base_model.parameters()).device
                new_head = new_head.to(device)
            self.head = new_head
            self.current_session = session_name
        else:
            raise ValueError(f"Session {session_name} not found in task_heads")
    
    def get_current_task_name(self):
        """获取当前任务名称"""
        if self.current_session and self.current_session in self.task_heads:
            return self.task_heads[self.current_session]['task_name']
        return None
    
    def save_task_heads(self, save_path: str):
        """保存所有任务的模型头"""
        task_heads_state = {}
        for session_name, task_info in self.task_heads.items():
            task_heads_state[session_name] = {
                'task_name': task_info['task_name'],
                'args': task_info['args'],
                'head_state_dict': task_info['head'].state_dict()
            }
        
        torch.save(task_heads_state, save_path)
    
    def load_task_heads(self, load_path: str, device: str, label_embedding_manager=None, logger=None):
        """
        加载历史任务的模型头
        """
        if not os.path.exists(load_path):
            if logger:
                logger.warning(f"Task heads file not found: {load_path}")
            return
        # 兼容PyTorch 2.6+安全机制，允许argparse.Namespace反序列化
        with torch.serialization.safe_globals([argparse.Namespace]):
            task_heads_state = torch.load(load_path, map_location=device)
            
            for session_name, task_info in task_heads_state.items():
                # 重新创建模型头
                try:
                    if hasattr(task_info['args'], 'use_label_embedding') and task_info['args'].use_label_embedding:
                        from models.task_heads.get_head_new import get_head
                    else:
                        from models.task_heads.get_head import get_head
                    
                    # 获取标签嵌入
                    label_emb = None
                    if label_embedding_manager:
                        label_emb = label_embedding_manager.get_embedding()
                    
                    # 为每个历史任务创建独立的模型头
                    # 对于MOE模型，需要使用base_model.base_model来获取原始的BaseMultimodalModel
                    if hasattr(self.base_model, 'base_model'):
                        # MOE模型的情况
                        base_model_for_head = self.base_model.base_model
                    else:
                        # 普通模型的情况
                        base_model_for_head = self.base_model
                    
                    head = get_head(task_info['task_name'], base_model_for_head, task_info['args'], label_emb=label_emb)
                    head.load_state_dict(task_info['head_state_dict'])
                    
                    # 确保模型头在正确的设备上
                    head = head.to(device)
                    
                    self.task_heads[session_name] = {
                        'head': head,
                        'task_name': task_info['task_name'],
                        'args': task_info['args']
                    }
                    
                    logger.info(f"Loaded task head for session {session_name} ({task_info['task_name']})")
                except Exception as e:
                    logger.warning(f"Warning: Failed to load task head for session {session_name}: {e}")
                    continue


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

def is_new_task(train_info, current_session_name):
    if train_info is None:
        return True
    return False

def create_model(args, device: str, label_embedding_manager: Optional[LabelEmbeddingManager] = None, logger=None):
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
            logger.info(f"Loading pretrained model from: {args.pretrained_model_path}")
            pretrained = torch.load(args.pretrained_model_path)
            model.base_model.load_state_dict(pretrained, strict=False)
        else:
            logger.info("No pretrained model loaded.")
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
        
        # 添加当前任务的模型头（MOE模型也需要任务头管理）
        full_model.add_task_head(args.session_name, args.task_name, head, args)
        
        # 设置当前任务为活动头
        full_model.set_active_head(args.session_name)
        
        if args.ddas:
            full_model.ddas = DDASRouter(
                feature_dim=base_model.text_hidden_size,
                threshold=args.ddas_threshold
            ).to(device)
        else:
            full_model.ddas = None
        
        # 加载预训练模型
        if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
            logger.info(f"Loading pretrained model from: {args.pretrained_model_path}")
            checkpoint = torch.load(args.pretrained_model_path, map_location=device)
            
            # 分离基础模型参数和任务头参数
            base_model_params = {}
            task_head_params = {}
            
            for key, value in checkpoint.items():
                if key.startswith('base_model.'):
                    base_model_params[key] = value
                elif key.startswith('head.'):
                    task_head_params[key] = value
                elif key.startswith('task_heads.'):
                    # 任务头参数，需要特殊处理
                    task_head_params[key] = value
            
            # 加载基础模型参数
            if base_model_params:
                moe_model.load_state_dict(base_model_params, strict=False)
            
            # 加载当前任务的模型头参数（只加载兼容的参数）
            if task_head_params:
                # 过滤掉与当前任务不匹配的模型头参数
                filtered_params = {}
                for key, value in task_head_params.items():
                    if key.startswith('head.'):
                        # 检查参数是否与当前任务兼容
                        current_head = full_model.head
                        param_name = key.replace('head.', '')
                        
                        # 递归查找参数
                        current_param = None
                        param_path = param_name.split('.')
                        current_obj = current_head
                        
                        try:
                            for part in param_path:
                                current_obj = getattr(current_obj, part)
                            current_param = current_obj
                        except AttributeError:
                            current_param = None
                        
                        if current_param is not None and hasattr(current_param, 'shape') and current_param.shape == value.shape:
                            filtered_params[key] = value
                            if logger:
                                logger.info(f"Loading compatible parameter: {key} - shape {value.shape}")
                        else:
                            if current_param is not None:
                                if logger:
                                    logger.info(f"Skipping incompatible parameter {key}: checkpoint shape {value.shape}, current shape {current_param.shape}")
                            else:
                                if logger:
                                    logger.info(f"Skipping parameter {key}: not found in current head")
                    else:
                        filtered_params[key] = value
                
                if filtered_params:
                    full_model.load_state_dict(filtered_params, strict=False)
                    if logger:
                        logger.info(f"Loaded {len(filtered_params)} compatible parameters")
            
            # 尝试加载历史任务头信息
            task_heads_path = args.pretrained_model_path.replace('.pt', '_task_heads.pt')
            if hasattr(full_model, 'load_task_heads') and os.path.exists(task_heads_path):
                logger.info(f"Loading task heads from: {task_heads_path}")
                full_model.load_task_heads(task_heads_path, device, label_embedding_manager, logger)
                
        else:
            logger.info("No pretrained model loaded.")
        
        return full_model
    
    elif args.clap4clip:
        from continual.clap4clip import CLAP4CLIP
        
        # 创建CLAP4CLIP模型
        clap4clip_model = CLAP4CLIP(
            text_model_name=args.text_model_name,
            image_model_name=args.image_model_name,
            num_labels=args.num_labels,
            dropout_prob=args.dropout_prob,
            adapter_size=getattr(args, 'adapter_size', 64),
            finetune_lambda=getattr(args, 'finetune_lambda', 0.1),
            temperature=getattr(args, 'temperature', 0.07)
        ).to(device)
        
        # 添加当前任务
        clap4clip_model.add_task(args.session_name, args.num_labels)
        clap4clip_model.set_current_task(args.session_name)
        
        # 加载预训练模型
        if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
            if logger:
                logger.info(f"Loading CLAP4CLIP from: {args.pretrained_model_path}")
            checkpoint = torch.load(args.pretrained_model_path, map_location=device)
            clap4clip_model.load_state_dict(checkpoint, strict=False)
        else:
            if logger:
                logger.info("No pretrained CLAP4CLIP model loaded.")
        
        return clap4clip_model
    
    else:
        base_model = BaseMultimodalModel(
            args.text_model_name,
            args.image_model_name,
            multimodal_fusion=args.fusion_strategy,
            num_heads=args.num_heads,
            mode=args.mode
        )

        # 使用标签嵌入
        label_emb = label_embedding_manager.get_embedding() if label_embedding_manager else None
        
        # 创建当前任务的模型头
        head = get_head(args.task_name, base_model, args, label_emb=label_emb)
        
        full_model = Full_Model(base_model, head, dropout_prob=args.dropout_prob).to(device)
        
        # 添加当前任务的模型头
        full_model.add_task_head(args.session_name, args.task_name, head, args)
        
        # 设置当前任务为活动头
        full_model.set_active_head(args.session_name)
        
        # 加载预训练模型
        if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
            logger.info(f"Loading pretrained model from: {args.pretrained_model_path}")
            checkpoint = torch.load(args.pretrained_model_path, map_location=device)
            
            # 分离基础模型参数和任务头参数
            base_model_params = {}
            task_head_params = {}
            
            for key, value in checkpoint.items():
                if key.startswith('base_model.'):
                    base_model_params[key] = value
                elif key.startswith('head.'):
                    task_head_params[key] = value
                elif key.startswith('task_heads.'):
                    # 任务头参数，需要特殊处理
                    task_head_params[key] = value
            
            # 加载基础模型参数
            if base_model_params:
                base_model.load_state_dict(base_model_params, strict=False)
            
            # 加载当前任务的模型头参数（只加载兼容的参数）
            if task_head_params:
                # 过滤掉与当前任务不匹配的模型头参数
                filtered_params = {}
                for key, value in task_head_params.items():
                    if key.startswith('head.'):
                        # 检查参数是否与当前任务兼容
                        current_head = full_model.head
                        param_name = key.replace('head.', '')
                        
                        # 递归查找参数
                        current_param = None
                        param_path = param_name.split('.')
                        current_obj = current_head
                        
                        try:
                            for part in param_path:
                                current_obj = getattr(current_obj, part)
                            current_param = current_obj
                        except AttributeError:
                            current_param = None
                        
                        if current_param is not None and hasattr(current_param, 'shape') and current_param.shape == value.shape:
                            filtered_params[key] = value
                            if logger:
                                logger.info(f"Loading compatible parameter: {key} - shape {value.shape}")
                        else:
                            if current_param is not None:
                                if logger:
                                    logger.info(f"Skipping incompatible parameter {key}: checkpoint shape {value.shape}, current shape {current_param.shape}")
                            else:
                                if logger:
                                    logger.info(f"Skipping parameter {key}: not found in current head")
                    else:
                        filtered_params[key] = value
                
                if filtered_params:
                    full_model.load_state_dict(filtered_params, strict=False)
                    if logger:
                        logger.info(f"Loaded {len(filtered_params)} compatible parameters")
            
            # 尝试加载历史任务头信息
            task_heads_path = args.pretrained_model_path.replace('.pt', '_task_heads.pt')
            if hasattr(full_model, 'load_task_heads') and os.path.exists(task_heads_path):
                logger.info(f"Loading task heads from: {task_heads_path}")
                full_model.load_task_heads(task_heads_path, device, label_embedding_manager, logger)
                
        else:
            logger.info("No pretrained model loaded.")
        
        return full_model


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
    session_info = {
        "session_name": args.session_name,
        "task_name": args.task_name,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": args.epochs,
        "details": {},
        "final_metrics": None,
        "args": vars(args),
    }
    
    # 如果是EWC或MyMethod，添加fisher_file路径
    if args.ewc == 1 or args.mymethod == 1:
        import os
        session_info["fisher_file"] = os.path.join(args.ewc_dir, f"{args.session_name}_fisher.pt")
    
    return session_info


def save_train_info(train_info: Dict[str, Any], train_info_path: str, logger=None):
    """保存训练信息"""
    with open(train_info_path, "w", encoding="utf-8") as f:
        json.dump(train_info, f, indent=2, ensure_ascii=False) 
    logger.info(f"Train info saved to: {train_info_path}")


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