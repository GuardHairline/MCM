# \continual\gem.py

import os
import torch
import torch.nn.functional as F
from utils.ensureFileExists import ensure_directory_exists
class GEMManager:
    """
    Implements a simplified GEM: store episodic memory per task, project gradients to avoid increasing past loss.
    """
    def __init__(self, model, memory_size=100, mem_dir="gem_memory"):
        self.model = model
        self.memory_size = memory_size
        self.mem_dir = mem_dir
        ensure_directory_exists(mem_dir)
        self.mem_data = {}  # task_name -> list of samples

    def register_task(self, task_session, dataset):
        """
        Randomly sample memory_size examples from dataset
        """
        mem_file = os.path.join(self.mem_dir, f"{task_session}_mem.pt")
        if os.path.exists(mem_file):
            # load persisted samples
            self.mem_data[task_session] = torch.load(mem_file)
        else:
            # sample and save later
            import random
            indices = random.sample(range(len(dataset)), min(self.memory_size, len(dataset)))
            samples = [dataset[i] for i in indices]
            self.mem_data[task_session] = samples
    def save_memory(self, task_session):
        # call after finishing training each task
        mem_file = os.path.join(self.mem_dir, f"{task_session}_mem.pt")
        torch.save(self.mem_data.get(task_session, []), mem_file)

    def project_gradients(self, grads, current_grad):
        """
        If gradient conflicts with memory gradient, project current_grad.
        Simplified: ensure dot(mem_grad, current_grad) >= 0
        """
        # 如果没有任何 memory buffer，跳过
        if not self.mem_data:
            return
        # collect memory grads
        mem_grads = []
        for task_session, samples in self.mem_data.items():
            # get gradient on memory samples
            self.model.zero_grad()
            batch = self._batch_collate(samples)
            outputs = self.model(**batch)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))
            loss.backward()

            vec = torch.cat([
                p.grad.contiguous().view(-1)
                for _, p in self.model.named_parameters()
                if p.grad is not None
            ])
            mem_grads.append(vec)

        # 如果所有 memory gradients 都为空，也跳过
        if not mem_grads:
            return

        mem_grad_vec = torch.stack(mem_grads).mean(dim=0)
        cur_grad_vec = torch.cat([g.view(-1) for g in current_grad])
        if torch.dot(cur_grad_vec, mem_grad_vec) < 0:
            # project: cur_grad = cur_grad - (dot(cur, mem)/|mem|^2)*mem
            proj = (torch.dot(cur_grad_vec, mem_grad_vec) / (mem_grad_vec.norm()**2)) * mem_grad_vec
            new = cur_grad_vec - proj
            # scatter back
            pointer = 0
            for _,p in self.model.named_parameters():
                if p.grad is not None:
                    numel = p.grad.numel()
                    p.grad.copy_( new[pointer:pointer+numel].view_as(p.grad) )
                    pointer += numel

    def _batch_collate(self, samples):
        # simplistic collate for dict of tensors
        batch = {}
        for k in samples[0]:
            batch[k] = torch.stack([s[k] for s in samples])
        return batch