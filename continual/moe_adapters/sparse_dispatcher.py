# continual/moe_adapters/sparse_dispatcher.py
"""
SparseDispatcher: 稀疏调度器
基于原论文实现，只将样本分发给gate>0的专家，避免不必要的计算。

参考：reference/MoE-Adapters4CL/cil/clip/model.py
"""
import torch
import torch.nn as nn


class SparseDispatcher:
    """
    稀疏专家调度器
    
    核心思想：
    - dispatch: 将输入样本分发给gate>0的专家（避免无效计算）
    - combine: 将专家输出加权合并（按gate权重）
    
    Example:
        gates = torch.tensor([[0.7, 0.3, 0.0], [0.0, 0.5, 0.5]])  # (B=2, E=3)
        dispatcher = SparseDispatcher(num_experts=3, gates=gates)
        
        # 分发
        expert_inputs = dispatcher.dispatch(inputs)  # 只有expert0,1,2收到对应样本
        
        # 专家处理
        expert_outputs = [expert(inp) for expert, inp in zip(experts, expert_inputs)]
        
        # 合并
        output = dispatcher.combine(expert_outputs)  # (B, D)
    """
    
    def __init__(self, num_experts: int, gates: torch.Tensor):
        """
        Args:
            num_experts: 专家总数
            gates: (batch_size, num_experts) gate权重，已经过softmax
        """
        self._gates = gates
        self._num_experts = num_experts
        
        # 找到所有非零gate的位置
        # nonzero返回 (N, 2)，每行是[batch_idx, expert_idx]
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        
        # 提取expert索引和batch索引
        _, self._expert_index = sorted_experts.split(1, dim=1)  # (N, 1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]  # (N,)
        
        # 计算每个专家获得的样本数
        self._part_sizes = (gates > 0).sum(0).tolist()  # list of E integers
        
        # 获取非零gate值
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)  # (N, 1)
    
    def dispatch(self, inp: torch.Tensor):
        """
        将输入分发给各个专家
        
        Args:
            inp: (batch_size, seq_len, hidden_dim) 或 (batch_size, hidden_dim)
        
        Returns:
            list of num_experts tensors，每个shape为(expert_batch_size, ...)
            注意：如果某专家gate全为0，对应tensor为空
        """
        # 选择gate>0的样本
        inp_exp = inp[self._batch_index]  # (N, seq_len, hidden_dim) or (N, hidden_dim)
        
        # 按专家分割
        return torch.split(inp_exp, self._part_sizes, dim=0)
    
    def combine(self, expert_outputs, multiply_by_gates: bool = True):
        """
        合并专家输出
        
        Args:
            expert_outputs: list of expert outputs, 每个shape为(expert_batch_size, ...)
            multiply_by_gates: 是否乘以gate权重
        
        Returns:
            combined: (batch_size, ...) 合并后的输出
        """
        # 拼接所有专家输出
        stitched = torch.cat(expert_outputs, 0)  # (N, ...)
        
        # 乘以gate权重
        if multiply_by_gates:
            # 扩展gate维度以匹配输出
            # stitched: (N, seq_len, hidden) or (N, hidden)
            # nonzero_gates: (N, 1)
            if stitched.dim() == 3:  # (N, seq_len, hidden)
                gates_expanded = self._nonzero_gates.unsqueeze(1)  # (N, 1, 1)
            else:  # (N, hidden)
                gates_expanded = self._nonzero_gates  # (N, 1)
            stitched = stitched * gates_expanded
        
        # 创建输出tensor
        batch_size = self._gates.size(0)
        output_shape = list(stitched.shape[1:])  # 去掉batch维
        zeros = torch.zeros(batch_size, *output_shape, 
                           device=stitched.device, dtype=stitched.dtype)
        
        # 将专家输出加到对应batch位置
        combined = zeros.index_add(0, self._batch_index, stitched)
        
        return combined
    
    def expert_to_gates(self):
        """
        返回每个专家对应的gate值（用于调试）
        
        Returns:
            list of num_experts tensors，每个包含该专家处理样本的gate值
        """
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


def test_sparse_dispatcher():
    """测试SparseDispatcher"""
    print("Testing SparseDispatcher...")
    
    # 测试1：基本功能
    gates = torch.tensor([
        [0.7, 0.3, 0.0],  # batch 0: expert 0,1
        [0.0, 0.5, 0.5],  # batch 1: expert 1,2
        [0.6, 0.2, 0.2],  # batch 2: expert 0,1,2
    ])
    
    inp = torch.randn(3, 768)  # (B=3, H=768)
    
    dispatcher = SparseDispatcher(num_experts=3, gates=gates)
    
    # 分发
    expert_inputs = dispatcher.dispatch(inp)
    print(f"Expert 0 receives {len(expert_inputs[0])} samples")  # 应该是2 (batch 0,2)
    print(f"Expert 1 receives {len(expert_inputs[1])} samples")  # 应该是3 (batch 0,1,2)
    print(f"Expert 2 receives {len(expert_inputs[2])} samples")  # 应该是2 (batch 1,2)
    
    # 模拟专家处理（恒等映射）
    expert_outputs = [inp for inp in expert_inputs]
    
    # 合并
    output = dispatcher.combine(expert_outputs, multiply_by_gates=True)
    print(f"Output shape: {output.shape}")  # 应该是(3, 768)
    
    # 测试2：序列输入
    inp_seq = torch.randn(3, 10, 768)  # (B=3, L=10, H=768)
    expert_inputs_seq = dispatcher.dispatch(inp_seq)
    expert_outputs_seq = [inp for inp in expert_inputs_seq]
    output_seq = dispatcher.combine(expert_outputs_seq, multiply_by_gates=True)
    print(f"Sequence output shape: {output_seq.shape}")  # 应该是(3, 10, 768)
    
    print("✓ SparseDispatcher test passed!")


if __name__ == "__main__":
    test_sparse_dispatcher()

