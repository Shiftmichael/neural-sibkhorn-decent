import math
from functools import partial
from typing import Optional

import numpy as np
import ot as pot
import torch
from .Sinkhorn_decent import SD
import matplotlib.pyplot as plt
import geomloss



class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an OT plan (wrt squared Euclidean
    cost) with different implementations of the plan calculation."""

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost=False,
        **kwargs,
    ):
        # ot_fn should take (a, b, M) as arguments where a, b are marginals and
        # M is a cost matrix
        if method == "exact":
            self.ot_fn = pot.emd
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.kwargs = kwargs

    def get_map(self, x0, x1):
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        return p

    def sample_map(self, pi, batch_size):
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size)
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1):
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0])
        return x0[i], x1[j]

    def sample_trajectory(self, X):
        # Assume X is [batch, times, dim]
        times = X.shape[1]
        pis = []
        for t in range(times - 1):
            pis.append(self.get_map(X[:, t], X[:, t + 1]))

        indices = [np.arange(X.shape[0])]
        for pi in pis:
            j = []
            for i in indices[-1]:
                j.append(np.random.choice(pi.shape[1], p=pi[i] / pi[i].sum()))
            indices.append(np.array(j))

        to_return = []
        for t in range(times):
            to_return.append(X[:, t][indices[t]])
        to_return = np.stack(to_return, axis=1)
        return to_return


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret


class SinkhornSampler:
    def __init__(
        self,
        backend = 'online',
        blur = 0.05,
        scaling = 0.95,
        device = 'cuda:0',
        SD_lr = 0.01,
        **kwargs
    ):
        self.kwargs = kwargs
        self.algorithm = SD(backend=backend, blur=blur, scaling=scaling)
        self.tgt_mass = None
        self.init_mass = None
        self.support = None
        self.T = 100
        self.SD_lr = SD_lr
        self.batch_size = 128
        self.particles = None
        self.device = device
        self.record_sinkdiv = []
        self.record_support = []
    
    def set_bs_t_particles(self, batch_size = None, T = None, particles = None):
        self.batch_size = batch_size
        self.T = T
        self.support = particles
        self.tgt_mass = torch.ones(batch_size, device = self.device) / batch_size
        self.init_mass = torch.ones(batch_size, device = self.device) / batch_size


    def sample_trajectory(self, x1):
        for step in range(self.T):
            # use Index sd_lr * exp((t - T) / (T / 4))  
            # lr = self.opts.SD_lr * math.exp((step - self.opts.T) / (self.opts.T / 4))
            lr = self.SD_lr
            self.algorithm.one_step_update(
                step_size = lr,
                init_particles = self.support,
                init_mass = self.init_mass,
                tgt_support = x1,
                tgt_mass = self.tgt_mass
            )
            support, _, vector = self.algorithm.get_state()
            # sinkhorn_divergence = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.9)
            # sinkhorn_distance = sinkhorn_divergence(support, x1)
            # print('sinkhorn', sinkhorn_distance)
            # plt.figure(figsize=(6, 6))
            # plt.scatter(support[:, 0].to('cpu'), support[:, 1].to('cpu'), s=10, alpha=0.8, c="black")
            # plt.xticks([])
            # plt.yticks([])
            # plt.savefig(f'./test/sinkhorn/sink_{step}')
            self.support = support
            self.record_sinkdiv.append(vector)   #[time, batch_size, 3*32*32]
            self.record_support.append(support)  #[time, batch_size, 3*32*32]
    @torch.no_grad()
    def sample_state(self):
        return torch.stack(self.record_sinkdiv), torch.stack(self.record_support)