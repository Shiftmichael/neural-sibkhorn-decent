import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import geomloss
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

imgdir = './test/lr0.02_t100_retrain50_same_time_training_torchdyn'
os.makedirs(imgdir, exist_ok=True)

blur = 0.05
scaling = 0.95
backend = 'online'
lr = 0.01
device = 'cuda:0'
t = 100
dim = 2
batch_size = 256
model = torch.load('models/8gaussian-moons/otcfm_v1.pt').to(device)
model.eval()
x1 = sample_moons(1024)
sinkhorn_divergence = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.9)
# for i in range(4):
#     x0 = sample_8gaussians(batch_size)
#     xt = x0
#     x_batch = []
#     x_batch.append(x0)
#     for step in range(t):
#         model_t = t / 100
#         t_batch = torch.ones(batch_size).type_as(x0)
#         vt = model(torch.cat([xt.to('cpu'), t_batch[:, None].to('cpu')], dim=-1))
#         xt = xt + lr * vt
#         x_batch.append(xt)
#     x_batch = torch.stack(x_batch).detach()
#     if i == 0:
#         x_all = x_batch
#     else:
#         x_all = torch.cat((x_all, x_batch), dim=1)

node = NeuralODE(torch_wrapper(model), solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
with torch.no_grad():
    traj = node.trajectory(
        sample_8gaussians(1024).to(device),
        t_span=torch.linspace(0, 1, 100).to(device),
    )
n = 2000
traj = traj.to('cpu')
for i in range(t):
    x_all_t = traj[i, :n].detach()
    # distribution_1 = torch.ones(x_all_t.shape[0])
    # distribution_2 = torch.ones(x_all_t.shape[0])
    # cost_matrix =(torch.cdist(x_all_t, x1) ** 2)
    # W2 = pot.emd2(distribution_1, distribution_2, cost_matrix)
    sinkhorn_distance = sinkhorn_divergence(x1, x_all_t)
    print(f'W2: {sinkhorn_distance}')


x = traj.numpy()
plt.figure(figsize=(6, 6))
plt.scatter(x[0, :n, 0], x[0, :n, 1], s=4, alpha=1, c="black")
plt.scatter(x[:, :n, 0], x[:, :n, 1], s=0.2, alpha=0.2, c="olive")
plt.scatter(x[t-1, :n, 0], x[t-1, :n, 1], s=4, alpha=1, c="blue")
plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
plt.xticks([])
plt.yticks([])
plt.savefig(f'{imgdir}/otcfm_v1_myevaluator_mlp_torchdyn_euler.png')