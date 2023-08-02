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

imgdir = './test/lr0.02_t100_retrain50_same_time_training_myevaluator'
os.makedirs(imgdir, exist_ok=True)

blur = 0.05
scaling = 0.95
backend = 'online'
lr = 0.02
device = 'cuda:0'
t = 100
dim = 2
batch_size = 256
model = torch.load('models/8gaussian-moons_sinkhorn/lr0.02_t100_retrain50_same_time_training/cfm_v1_800.pt').to('cpu')
model.eval()
x1 = sample_moons(1024)
sinkhorn_divergence = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.9)
for i in range(4):
    x0 = sample_8gaussians(batch_size)
    xt = x0
    x_batch = []
    x_batch.append(x0)
    for step in range(t):
        model_t = t / 100
        t_batch = torch.ones(batch_size).type_as(x0)
        vt = model(torch.cat([xt.to('cpu'), t_batch[:, None].to('cpu')], dim=-1))
        xt = xt + lr * vt
        x_batch.append(xt)
    x_batch = torch.stack(x_batch).detach()
    if i == 0:
        x_all = x_batch
    else:
        x_all = torch.cat((x_all, x_batch), dim=1)

for i in range(t):
    x_all_t = x_all[i].detach()
    # distribution_1 = torch.ones(x_all_t.shape[0])
    # distribution_2 = torch.ones(x_all_t.shape[0])
    # cost_matrix =(torch.cdist(x_all_t, x1) ** 2)
    # W2 = pot.emd2(distribution_1, distribution_2, cost_matrix)
    sinkhorn_distance = sinkhorn_divergence(x1, x_all_t)
    print(f'W2: {sinkhorn_distance}')


x = x_all.numpy()
plt.figure(figsize=(6, 6))
plt.scatter(x[0, :, 0], x[0, :, 1], s=4, alpha=1, c="black")
plt.scatter(x[:, :, 0], x[:, :, 1], s=0.2, alpha=0.2, c="olive")
plt.scatter(x[t, :, 0], x[t, :, 1], s=4, alpha=1, c="blue")
plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
plt.xticks([])
plt.yticks([])
plt.savefig(f'{imgdir}/sink_dataext_retrain50_myevaluator_800.png')
