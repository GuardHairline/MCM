# \continual\mas.py

import torch
class MASRegularizer:
    """
    Memory Aware Synapses (MAS) estimates parameter importance via sensitivity of output L2 norm.
    After each task, omega is computed and used as a regularizer for future tasks.
    """
    def __init__(self, model, epsilon=1e-3):
        self.model = model
        self.epsilon = epsilon
        self.omega = {n: torch.zeros_like(p) for n,p in model.named_parameters()}
        self.prev_params = {n: p.clone().detach() for n,p in model.named_parameters()}

    @torch.no_grad()
    def compute_importance(self, data_loader, device):
        """
        Estimate importance by accumulating gradients of output L2 norm.
        """
        # zero omega accumulator
        for n in self.omega: self.omega[n].zero_()
        self.model.eval()
        for batch in data_loader:
            inputs = {k:batch[k].to(device) for k in ['input_ids','attention_mask','image_tensor'] if k in batch}
            tok = batch.get('token_type_ids', None)
            if tok is not None: inputs['token_type_ids']=tok.to(device)

            self.model.zero_grad()
            # forward and compute output norm
            logits = self.model(**inputs)
            score = (logits**2).sum()
            score.backward()
            # accumulate |grad|
            for n,p in self.model.named_parameters():
                if p.grad is not None:
                    self.omega[n] += p.grad.abs()
        # normalize
        for n in self.omega:
            self.omega[n] = self.omega[n] / len(data_loader)
        # save params snapshot
        self.prev_params = {n: p.clone().detach() for n,p in self.model.named_parameters()}

    def penalty(self):
        """
        Compute MAS penalty: sum omega * (p - p_prev)^2
        """
        loss = 0
        for n,p in self.model.named_parameters():
            loss += (self.omega[n].to(p.device) * (p - self.prev_params[n].to(p.device))**2).sum()
        return self.epsilon * loss
