# -------------------------------------------------------------------
# File: continual/si.py
# Description: Synaptic Intelligence (SI) module
# -------------------------------------------------------------------
import torch

class SynapticIntelligence:
    """
    Implements Synaptic Intelligence: accumulate importance online.
    """
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon
        # Omega: importance of each parameter
        self.omega = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        # Previous parameters snapshot
        self.prev_params = {n: p.clone().detach() for n, p in model.named_parameters()}
        # Accumulator for path integral
        self._accum = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    def accumulate(self, grads):
        # grads: dict of parameter gradients after each update step
        for n, g in grads.items():
            self._accum[n] += (-g * (self.model.state_dict()[n] - self.prev_params[n]))
        # update prev_params
        for n, p in self.model.named_parameters():
            self.prev_params[n] = p.clone().detach()

    def update_omega(self):
        # Compute final omega for each parameter
        for n in self.omega:
            self.omega[n] += self._accum[n] / ( (self.model.state_dict()[n] - self.prev_params[n]).pow(2) + 1e-10 )
        # reset accumulator
        self._accum = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

    def penalty(self):
        # L2 penalty weighted by omega
        loss = 0
        for n, p in self.model.named_parameters():
            loss += (self.omega[n] * (p - self.prev_params[n]).pow(2)).sum()
        return self.epsilon * loss
