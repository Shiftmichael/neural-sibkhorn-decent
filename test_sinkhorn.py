import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

savedir = "models/8gaussian-moons_sinkhorn"
os.makedirs(savedir, exist_ok=True)

blur = 0.05
scaling = 0.95
backend = 'online'
lr = 0.06
device = 'cuda:3'
t = 100
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters())
FM = SinkhornFlowMatcher(blur = blur, scaling = scaling, backend = backend, lr = lr, device = device)

start = time.time()
for k in range(5000):
    optimizer.zero_grad()

    x0 = sample_8gaussians(batch_size).to(device)
    x1 = sample_moons(batch_size).to(device)

    t_train, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)
    FM.clear_all()

    vt = model(torch.cat([xt.to('cpu'), t_train[:, None].to('cpu')], dim=-1))
    loss = torch.mean((vt - ut.to('cpu')) ** 2)

    loss.backward()
    optimizer.step()

    if (k + 1) % 100 == 0 or k == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end
        node = NeuralODE(
            torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        with torch.no_grad():
            traj = node.trajectory(
                sample_8gaussians(1024),
                t_span=torch.linspace(0, 1, 100),
            )
            n = 2000
            plt.figure(figsize=(6, 6))
            plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
            plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="red")
            plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
            plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'./test/sink_{k}_200')

torch.save(model, f"{savedir}/cfm_v1.pt")