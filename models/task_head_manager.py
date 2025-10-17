# models/task_head_manager.py
"""
统一的任务头管理器
解决问题：
1. 预创建所有任务头（包括未来任务）
2. TAM-CL和其他模型逻辑不一致
3. head切换缺少错误处理
4. head状态验证缺失
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaskHeadInfo:
    """任务头信息"""
    session_name: str
    task_name: str
    head: nn.Module
    args: Any
    created_at: str
    is_frozen: bool = False
    
    def freeze(self):
        """冻结任务头参数"""
        for param in self.head.parameters():
            param.requires_grad = False
        self.is_frozen = True
        
    def unfreeze(self):
        """解冻任务头参数"""
        for param in self.head.parameters():
            param.requires_grad = True
        self.is_frozen = False


class TaskHeadManager:
    """
    统一的任务头管理器
    
    特性：
    1. 延迟创建：只在需要时创建head
    2. 状态验证：确保head正确切换
    3. 统一接口：支持所有模型类型
    4. 错误处理：提供清晰的错误信息
    """
    
    def __init__(self, base_model, label_embedding_manager=None, device='cuda'):
        self.base_model = base_model
        self.label_embedding_manager = label_embedding_manager
        self.device = device
        
        # 存储所有任务头
        self._task_heads: Dict[str, TaskHeadInfo] = {}
        
        # 当前活动的任务头
        self._current_session: Optional[str] = None
        self._current_head: Optional[nn.Module] = None
        
        # 统计信息
        self._head_usage_count: Dict[str, int] = {}
        
    def register_head(self, session_name: str, task_name: str, 
                     head: nn.Module, args: Any, freeze: bool = False) -> bool:
        """
        注册一个任务头
        
        Args:
            session_name: 会话名称（唯一标识）
            task_name: 任务名称
            head: 任务头模块
            args: 任务参数
            freeze: 是否立即冻结
            
        Returns:
            是否注册成功
        """
        if session_name in self._task_heads:
            logger.warning(f"Session '{session_name}' already registered, skipping")
            return False
        
        # 确保head在正确的设备上
        head = head.to(self.device)
        
        # 创建头信息
        import time
        head_info = TaskHeadInfo(
            session_name=session_name,
            task_name=task_name,
            head=head,
            args=args,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            is_frozen=freeze
        )
        
        if freeze:
            head_info.freeze()
        
        self._task_heads[session_name] = head_info
        self._head_usage_count[session_name] = 0
        
        logger.info(f"Registered task head: {session_name} ({task_name}), frozen={freeze}")
        return True
    
    def create_and_register_head(self, session_name: str, task_name: str, 
                                args: Any, use_label_embedding: bool = False) -> Optional[nn.Module]:
        """
        创建并注册任务头（延迟创建模式）
        
        Args:
            session_name: 会话名称
            task_name: 任务名称
            args: 任务参数
            use_label_embedding: 是否使用标签嵌入
            
        Returns:
            创建的任务头，如果失败返回None
        """
        if session_name in self._task_heads:
            logger.info(f"Task head '{session_name}' already exists, reusing")
            return self._task_heads[session_name].head
        
        try:
            # 选择合适的head创建函数
            if use_label_embedding:
                from models.task_heads.get_head_new import get_head
            else:
                from models.task_heads.get_head import get_head
            
            # 获取标签嵌入（如果需要）
            label_emb = None
            if self.label_embedding_manager and use_label_embedding:
                label_emb = self.label_embedding_manager.get_embedding()
            
            # 创建head
            head = get_head(task_name, self.base_model, args, label_emb=label_emb)
            
            # 注册
            self.register_head(session_name, task_name, head, args)
            
            return head
            
        except Exception as e:
            logger.error(f"Failed to create head for '{session_name}': {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def set_active_head(self, session_name: str, strict: bool = True) -> bool:
        """
        设置活动任务头
        
        Args:
            session_name: 会话名称
            strict: 是否严格模式（找不到head时报错）
            
        Returns:
            是否切换成功
        """
        if session_name not in self._task_heads:
            msg = f"Session '{session_name}' not found in registered heads: {list(self._task_heads.keys())}"
            if strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False
        
        # 切换head
        head_info = self._task_heads[session_name]
        self._current_session = session_name
        self._current_head = head_info.head
        
        # 更新使用计数
        self._head_usage_count[session_name] += 1
        
        logger.debug(f"Switched to head: {session_name} ({head_info.task_name})")
        return True
    
    def get_current_head(self) -> Optional[nn.Module]:
        """获取当前活动的任务头"""
        return self._current_head
    
    def get_current_session(self) -> Optional[str]:
        """获取当前活动的会话名称"""
        return self._current_session
    
    def get_head(self, session_name: str) -> Optional[nn.Module]:
        """获取指定会话的任务头"""
        if session_name not in self._task_heads:
            return None
        return self._task_heads[session_name].head
    
    def get_task_name(self, session_name: str) -> Optional[str]:
        """获取指定会话的任务名称"""
        if session_name not in self._task_heads:
            return None
        return self._task_heads[session_name].task_name
    
    def has_head(self, session_name: str) -> bool:
        """检查是否存在指定会话的任务头"""
        return session_name in self._task_heads
    
    def remove_head(self, session_name: str) -> bool:
        """
        移除指定的任务头
        
        Args:
            session_name: 会话名称
            
        Returns:
            是否移除成功
        """
        if session_name not in self._task_heads:
            logger.warning(f"Cannot remove non-existent head: {session_name}")
            return False
        
        # 如果正在使用这个head，清除当前状态
        if self._current_session == session_name:
            self._current_session = None
            self._current_head = None
        
        # 删除head
        del self._task_heads[session_name]
        if session_name in self._head_usage_count:
            del self._head_usage_count[session_name]
        
        logger.info(f"Removed task head: {session_name}")
        return True
    
    def freeze_head(self, session_name: str) -> bool:
        """冻结指定任务头"""
        if session_name not in self._task_heads:
            logger.warning(f"Cannot freeze non-existent head: {session_name}")
            return False
        
        self._task_heads[session_name].freeze()
        logger.info(f"Frozen task head: {session_name}")
        return True
    
    def freeze_all_except(self, session_name: str) -> int:
        """冻结除指定会话外的所有任务头"""
        count = 0
        for sess_name in self._task_heads:
            if sess_name != session_name:
                if self.freeze_head(sess_name):
                    count += 1
        logger.info(f"Frozen {count} task heads (except {session_name})")
        return count
    
    def unfreeze_head(self, session_name: str) -> bool:
        """解冻指定任务头"""
        if session_name not in self._task_heads:
            logger.warning(f"Cannot unfreeze non-existent head: {session_name}")
            return False
        
        self._task_heads[session_name].unfreeze()
        logger.info(f"Unfrozen task head: {session_name}")
        return True
    
    def get_all_sessions(self) -> List[str]:
        """获取所有已注册会话的名称"""
        return list(self._task_heads.keys())
    
    def get_head_count(self) -> int:
        """获取已注册任务头的数量"""
        return len(self._task_heads)
    
    def save_heads(self, save_path: str) -> bool:
        """
        保存所有任务头
        
        Args:
            save_path: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            heads_state = {}
            for session_name, head_info in self._task_heads.items():
                heads_state[session_name] = {
                    'task_name': head_info.task_name,
                    'args': head_info.args,
                    'head_state_dict': head_info.head.state_dict(),
                    'created_at': head_info.created_at,
                    'is_frozen': head_info.is_frozen
                }
            
            torch.save(heads_state, save_path)
            logger.info(f"Saved {len(heads_state)} task heads to: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save task heads: {e}")
            return False
    
    def load_heads(self, load_path: str, strict: bool = False) -> int:
        """
        加载任务头
        
        Args:
            load_path: 加载路径
            strict: 是否严格模式（加载失败时报错）
            
        Returns:
            成功加载的任务头数量
        """
        if not os.path.exists(load_path):
            msg = f"Task heads file not found: {load_path}"
            if strict:
                raise FileNotFoundError(msg)
            else:
                logger.warning(msg)
                return 0
        
        try:
            heads_state = torch.load(load_path, map_location=self.device)
            loaded_count = 0
            
            for session_name, head_data in heads_state.items():
                try:
                    # 重新创建任务头
                    task_name = head_data['task_name']
                    args = head_data['args']
                    use_label_embedding = getattr(args, 'use_label_embedding', False)
                    
                    # 创建head
                    head = self.create_and_register_head(
                        session_name, task_name, args, use_label_embedding
                    )
                    
                    if head is not None:
                        # 加载参数
                        head.load_state_dict(head_data['head_state_dict'])
                        
                        # 恢复冻结状态
                        if head_data.get('is_frozen', False):
                            self.freeze_head(session_name)
                        
                        loaded_count += 1
                        logger.info(f"Loaded task head: {session_name} ({task_name})")
                    else:
                        logger.warning(f"Failed to create head for: {session_name}")
                        
                except Exception as e:
                    msg = f"Failed to load head '{session_name}': {e}"
                    if strict:
                        raise RuntimeError(msg)
                    else:
                        logger.warning(msg)
                        continue
            
            logger.info(f"Successfully loaded {loaded_count}/{len(heads_state)} task heads")
            return loaded_count
            
        except Exception as e:
            msg = f"Failed to load task heads from {load_path}: {e}"
            if strict:
                raise RuntimeError(msg)
            else:
                logger.error(msg)
                return 0
    
    def print_summary(self):
        """打印任务头管理器摘要"""
        print("="*80)
        print("Task Head Manager Summary")
        print("="*80)
        print(f"Total registered heads: {len(self._task_heads)}")
        print(f"Current active session: {self._current_session}")
        print(f"Device: {self.device}")
        print("\nRegistered heads:")
        
        for session_name, head_info in self._task_heads.items():
            is_current = "✓" if session_name == self._current_session else " "
            frozen = "🔒" if head_info.is_frozen else "🔓"
            usage = self._head_usage_count.get(session_name, 0)
            
            print(f"  [{is_current}] {frozen} {session_name}")
            print(f"      Task: {head_info.task_name}")
            print(f"      Created: {head_info.created_at}")
            print(f"      Usage count: {usage}")
        
        print("="*80)
    
    def validate_head(self, session_name: str) -> tuple[bool, str]:
        """
        验证任务头是否正常
        
        Returns:
            (is_valid, error_message)
        """
        if session_name not in self._task_heads:
            return False, f"Head not found: {session_name}"
        
        head_info = self._task_heads[session_name]
        head = head_info.head
        
        # 检查head是否在正确的设备上
        try:
            first_param = next(head.parameters())
            if str(first_param.device) != str(self.device):
                return False, f"Head on wrong device: {first_param.device} (expected {self.device})"
        except StopIteration:
            return False, "Head has no parameters"
        
        # 检查head是否可以前向传播
        # 这里可以添加更多验证逻辑
        
        return True, "OK"


def create_task_head_manager(base_model, label_embedding_manager=None, 
                             device='cuda') -> TaskHeadManager:
    """
    创建任务头管理器的工厂函数
    
    Args:
        base_model: 基础模型
        label_embedding_manager: 标签嵌入管理器
        device: 设备
        
    Returns:
        任务头管理器实例
    """
    return TaskHeadManager(base_model, label_embedding_manager, device)

