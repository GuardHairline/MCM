# continual/gem.py
import os
import torch
import torch.nn.functional as F
from typing import Dict, List
from utils.ensureFileExists import ensure_directory_exists

class GEMManager:
    """
    Gradient Episodic Memory (GEM) 管理器。

    该类负责为每个任务保存小规模记忆样本并在每次反向传播时
    检查当前梯度是否与记忆中的梯度冲突。如有冲突，则将当前梯度
    投影到与记忆梯度一致的半空间以减少灾难遗忘。
    """

    def __init__(self, model, memory_size=100, mem_dir="gem_memory", device=None):
        """
        Args:
            model: 完整的 Full_Model，其中包含 base_model 和多个 task head。
            memory_size: 每个任务存储的记忆样本数。
            mem_dir: 记忆样本保存目录。
            device: 梯度计算所用的设备（'cpu' 或 'cuda'）。若为 None 则自动从模型获取。
        """
        self.model = model
        self.memory_size = memory_size
        self.mem_dir = mem_dir
        ensure_directory_exists(mem_dir)
        self.mem_data: Dict[str, List[dict]] = {}  # task_name -> list of samples

        # 设备设置：如果未显式传入，则从模型中推断
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 延迟导入，避免循环引用；若导入失败则使用简单判断
        try:
            from modules.train_utils import is_sequence_task
            self.is_sequence_task = is_sequence_task
        except Exception:
            self.is_sequence_task = lambda name: name in ["mate", "mner", "mabsa"]

    def register_task(self, task_name: str, dataset) -> None:
        """
        为一个新任务注册记忆样本。若磁盘上已有保存的记忆样本，则直接加载；
        否则随机采样 memory_size 个样本。

        Args:
            task_name: 任务名称（如 'mabsa'）。
            dataset: 该任务的训练集对象。
        """
        mem_file = os.path.join(self.mem_dir, f"{task_name}_mem.pt")
        if os.path.exists(mem_file):
            try:
                self.mem_data[task_name] = torch.load(mem_file)
                return
            except Exception:
                # 若加载失败则重新采样
                pass
        # 随机采样
        import random
        num = min(self.memory_size, len(dataset))
        indices = random.sample(range(len(dataset)), num)
        samples = [dataset[i] for i in indices]
        self.mem_data[task_name] = samples

    def save_memory(self, task_name: str) -> None:
        """
        将指定任务的记忆样本保存到磁盘。
        """
        mem_file = os.path.join(self.mem_dir, f"{task_name}_mem.pt")
        torch.save(self.mem_data.get(task_name, []), mem_file)

    def project_gradients(self, grads, current_grad) -> None:
        """
        根据记忆样本的梯度投影当前梯度，避免干扰过去任务。

        Args:
            grads: 兼容旧接口，此处不使用，可为 None。
            current_grad: 一个包含当前模型各参数梯度的列表，
                通常在调用 loss.backward() 后通过
                `[p.grad for p in model.parameters() if p.grad is not None]` 获取。
        """
        if not self.mem_data:
            return  # 若没有记忆样本，直接返回

        mem_grads = []
        # 遍历所有历史任务，计算记忆梯度
        for task_name, samples in self.mem_data.items():
            if not samples:
                continue
            # 判断该任务是序列任务还是句级任务
            is_seq = self.is_sequence_task(task_name)

            # 将记忆样本批量化并移动到目标设备
            batch = self._batch_collate(samples)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            image_tensor = batch['image_tensor'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.model.zero_grad()

            # 先通过 base_model 获取特征
            fused_feat = self.model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                image_tensor=image_tensor,
                return_sequence=is_seq
            )

            # 再取出对应任务的 head；若没有找到则用当前 head
            if hasattr(self.model, 'task_heads') and task_name in self.model.task_heads:
                head = self.model.task_heads[task_name]['head']
            else:
                head = self.model.head
            head = head.to(self.device)

            logits = head(fused_feat)

            # 根据任务类型计算损失
            if is_seq:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100
                )
            else:
                loss = F.cross_entropy(logits, labels)

            loss.backward()

            # 将各参数梯度拼接成一维向量
            vec = torch.cat([
                p.grad.detach().contiguous().view(-1)
                for _, p in self.model.named_parameters()
                if p.grad is not None
            ])
            mem_grads.append(vec)

        if not mem_grads:
            return

        # 计算所有记忆梯度的平均值
        mem_grad_vec = torch.stack(mem_grads).mean(dim=0)
        # 将当前梯度展开为向量
        cur_grad_vec = torch.cat([g.detach().view(-1) for g in current_grad])

        # 若点积为负，则投影到与 mem_grad_vec 同向的半空间
        if torch.dot(cur_grad_vec, mem_grad_vec) < 0:
            proj = (torch.dot(cur_grad_vec, mem_grad_vec) /
                    (mem_grad_vec.norm() ** 2)) * mem_grad_vec
            new_vec = cur_grad_vec - proj

            # 将投影后的梯度写回各参数
            pointer = 0
            for _, p in self.model.named_parameters():
                if p.grad is not None:
                    numel = p.grad.numel()
                    p.grad.copy_(new_vec[pointer: pointer + numel].view_as(p.grad))
                    pointer += numel

    def _batch_collate(self, samples: List[dict]) -> dict:
        """
        将记忆样本列表组装成批量。不会移动设备。

        Args:
            samples: 单个记忆样本字典的列表。

        Returns:
            批量字典，每个键对应一个形状为 (batch_size, ...) 的张量。
        """
        assert len(samples) > 0
        batch = {}
        # 只对 Tensor 类型的字段进行堆叠
        for key, value in samples[0].items():
            if isinstance(value, torch.Tensor):
                batch[key] = torch.stack([s[key] for s in samples], dim=0)
        return batch
