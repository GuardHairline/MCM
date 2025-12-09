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
import argparse
try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None
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
    head_key: Optional[str] = None
    
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
        
        # [核心结构]
        # 1. 物理头存储: head_key -> TaskHeadInfo
        #    这里存储真实的模型对象和元数据
        self._task_heads: Dict[str, TaskHeadInfo] = {}  # key=head_key
        # 2. 映射表: session_name -> head_key
        #    这是对外接口（Session）到内部存储（Physical Head）的桥梁
        self._session_to_headkey: Dict[str, str] = {}   # session_name -> head_key
        
        # 当前活动的任务头
        self._current_session: Optional[str] = None
        self._current_head: Optional[nn.Module] = None
        
        # 统计信息
        self._head_usage_count: Dict[str, int] = {}
    def _resolve_key(self, session_name: str) -> Optional[str]:
        """[内部辅助] 将 session_name 解析为 head_key"""
        return self._session_to_headkey.get(session_name)
    def register_head(self, session_name: str, task_name: str, 
                     head: nn.Module, args: Any, freeze: bool = False,
                     head_key: Optional[str] = None) -> bool:
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
        # 1. 确定 head_key
        # 如果未提供，默认使用 session_name 作为 key (即不共享)
        key = head_key or session_name

        # 2. 建立映射关系 (session -> key)
        if session_name in self._session_to_headkey:
            logger.warning(f"Session '{session_name}' already registered (mapped to '{self._session_to_headkey[session_name]}'). Overwriting mapping.")
        self._session_to_headkey[session_name] = key

        # 3. 如果该物理头已存在，说明是复用
        if key in self._task_heads:
            logger.info(f"♻️ Head '{key}' already exists. Session '{session_name}' will reuse it.")
            # 注意：这里我们不更新已存在的 HeadInfo (如 args)，保持创建时的状态
            # 但我们需要更新 freeze 状态吗？通常遵循创建者的设定。
            return True
        
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
            is_frozen=freeze,
            head_key=key
        )
        
        if freeze:
            head_info.freeze()
        
        self._task_heads[key] = head_info
        self._head_usage_count[key] = 0
        
        logger.info(f"Registered task head: {key} ({task_name}), frozen={freeze}, session='{session_name}'")
        return True
    
    def create_and_register_head(self, session_name: str, task_name: str, 
                                args: Any, use_label_embedding: bool = False,
                                head_key: Optional[str] = None) -> Optional[nn.Module]:
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
        # 1. 确定 Key
        key = head_key or session_name

        # 2. 检查复用
        if key in self._task_heads:
            logger.info(f"Task head '{key}' already exists, reusing (session '{session_name}')")
            self._session_to_headkey[session_name] = key
            return self._task_heads[key].head
        # 3. 创建新 Head
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
            self.register_head(session_name, task_name, head, args, head_key=head_key)
            
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
        # 1. 解析 Key
        key = self._resolve_key(session_name)
        
        # 2. 检查是否存在
        if not key or key not in self._task_heads:
            msg = f"Session '{session_name}' not mapped to any registered head."
            if strict: raise ValueError(msg)
            logger.warning(msg)
            return False
        
        # 3. 切换
        head_info = self._task_heads[key]
        self._current_session = session_name
        self._current_head = head_info.head
        
        # 更新使用计数
        self._head_usage_count[key] += 1
        
        logger.debug(f"Switched to head: {key} (Session: {session_name})")        
        return True
    
    def get_current_head(self) -> Optional[nn.Module]:
        """获取当前活动的任务头"""
        return self._current_head
    
    def get_current_session(self) -> Optional[str]:
        """获取当前活动的会话名称"""
        return self._current_session
    
    def get_head(self, session_name: str) -> Optional[nn.Module]:
        key = self._resolve_key(session_name)
        if key and key in self._task_heads:
            return self._task_heads[key].head
        return None
    
    def get_task_name(self, session_name: str) -> Optional[str]:
        key = self._resolve_key(session_name)
        if key and key in self._task_heads:
            return self._task_heads[key].task_name
        return None
    
    def has_head(self, session_name: str) -> bool:
        """检查 session 是否有对应的 head"""
        key = self._resolve_key(session_name)
        return key is not None and key in self._task_heads
    
    def remove_head(self, session_name: str) -> bool:
        """
        移除指定的任务头
        
        Args:
            session_name: 会话名称
            
        Returns:
            是否移除成功
        """
        if session_name not in self._session_to_headkey:
            return False
        
        key = self._session_to_headkey[session_name]
        
        # 删除映射
        del self._session_to_headkey[session_name]
        
        # 检查是否还有其他 session 引用这个 key
        has_refs = any(k == key for k in self._session_to_headkey.values())
        
        if not has_refs:
            # 没有引用了，彻底删除物理头
            if key in self._task_heads:
                del self._task_heads[key]
            if key in self._head_usage_count:
                del self._head_usage_count[key]
            logger.info(f"Removed physical head '{key}' (was linked to '{session_name}')")
        else:
            logger.info(f"Unmapped session '{session_name}' from head '{key}' (Head still active)")
            
        if self._current_session == session_name:
            self._current_session = None
            self._current_head = None
            
        return True
    
    def freeze_head(self, session_name: str) -> bool:
        key = self._resolve_key(session_name)
        if key and key in self._task_heads:
            self._task_heads[key].freeze()
            return True
        return False
    
    def freeze_all_except(self, session_name: str) -> int:
        """冻结除指定 Session 所用 Head 之外的所有物理头"""
        target_key = self._resolve_key(session_name)
        count = 0
        for key, info in self._task_heads.items():
            if key != target_key:
                info.freeze()
                count += 1
            else:
                info.unfreeze() # 确保目标是解冻的
        return count
    
    def unfreeze_head(self, session_name: str) -> bool:
        key = self._resolve_key(session_name)
        if key and key in self._task_heads:
            self._task_heads[key].unfreeze()
            return True
        return False
    
    def get_all_sessions(self) -> List[str]:
        return list(self._session_to_headkey.keys())
    
    def get_head_count(self) -> int:
        """返回物理头的数量"""
        return len(self._task_heads)
    
    # --- 保存与加载 ---
    
    def save_heads(self, save_path: str) -> bool:
        try:
            state = {
                # 1. 物理头数据 (去重存储)
                'physical_heads': {
                    key: {
                        'state_dict': info.head.state_dict(),
                        'task_name': info.task_name,
                        'args': info.args,
                        'created_at': info.created_at,
                        'is_frozen': info.is_frozen
                    }
                    for key, info in self._task_heads.items()
                },
                # 2. 映射关系
                'session_mapping': self._session_to_headkey
            }
            torch.save(state, save_path)
            logger.info(f"Saved {len(self._task_heads)} heads and {len(self._session_to_headkey)} sessions to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False
    def _torch_load_with_weights(self, load_path, map_location):
        """
        兼容不同 PyTorch 版本的加载逻辑
        """
        # 1. 尝试注册安全全局变量 (仅新版本有效)
        if add_safe_globals is not None:
            try:
                add_safe_globals([argparse.Namespace])
            except Exception:
                pass
                
        # 2. 尝试使用 weights_only=False (新版本默认 True)
        try:
            return torch.load(load_path, map_location=map_location, weights_only=False)
        except TypeError:
            # 老版本 PyTorch 不支持 weights_only 参数
            return torch.load(load_path, map_location=map_location)
    def load_heads(self, load_path: str, strict: bool = False) -> int:
        if not os.path.exists(load_path):
            if strict: raise FileNotFoundError(load_path)
            return 0
        
        try:
            state = self._torch_load_with_weights(load_path, map_location=self.device)
            
            # 检查格式
            if isinstance(state, dict) and 'physical_heads' in state:
                # 新版格式
                phy_heads = state['physical_heads']
                mapping = state.get('session_mapping', {})
            else:
                # 旧版格式 (直接是 session -> info)
                # 尝试自动升级
                phy_heads = {}
                mapping = {}
                for k, v in state.items():
                    phy_heads[k] = {
                        'state_dict': v['head_state_dict'],
                        'task_name': v['task_name'],
                        'args': v['args']
                    }
                    mapping[k] = k
            
            # 恢复数据
            loaded = 0
            self._session_to_headkey.update(mapping)
            
            for key, data in phy_heads.items():
                try:
                    # 重新创建 head
                    self.create_and_register_head(
                        session_name=key, # 临时用 key 当 session
                        task_name=data['task_name'],
                        args=data['args'],
                        use_label_embedding=getattr(data['args'], 'use_label_embedding', False),
                        head_key=key # 强制指定 Key
                    )
                    
                    # 加载参数
                    if key in self._task_heads:
                        self._task_heads[key].head.load_state_dict(data['state_dict'])
                        
                        # 恢复冻结状态
                        if data.get('is_frozen', False):
                            self._task_heads[key].freeze()
                            
                        loaded += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to restore head '{key}': {e}")
            
            logger.info(f"Restored {loaded} physical heads.")
            return loaded
            
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return 0
            
    def print_summary(self):
        print(f"\n[HeadManager] Physical Heads: {len(self._task_heads)} | Mapped Sessions: {len(self._session_to_headkey)}")
        for key, info in self._task_heads.items():
            sessions = [s for s, k in self._session_to_headkey.items() if k == key]
            print(f"  - Key: {key:<20} | Task: {info.task_name:<5} | Used by: {sessions}")

def create_task_head_manager(base_model, label_embedding_manager=None, device='cuda') -> TaskHeadManager:
    return TaskHeadManager(base_model, label_embedding_manager, device)