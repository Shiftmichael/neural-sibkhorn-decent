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
model = torch.load('models/8gaussian-moons_sinkhorn/lr0.02_t100_retrain50_same_time_training/cfm_v1_200.pt').to('cpu')
model.eval()
x0 = sample_8gaussians(256)
xt = x0
x = []
x.append(x0)
for step in range(t):
    model_t = t / 100
    t_batch = torch.ones(batch_size).type_as(x0)
    vt = model(torch.cat([xt.to('cpu'), t_batch[:, None].to('cpu')], dim=-1))
    xt = xt + lr * vt
    x.append(xt)
x = torch.stack(x).detach().numpy()
plt.figure(figsize=(6, 6))
plt.scatter(x[0, :, 0], x[0, :, 1], s=4, alpha=1, c="black")
plt.scatter(x[:, :, 0], x[:, :, 1], s=0.2, alpha=0.2, c="olive")
plt.scatter(x[t, :, 0], x[t, :, 1], s=4, alpha=1, c="blue")
plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
plt.xticks([])
plt.yticks([])
plt.savefig(f'{imgdir}/sinkhorn.png')
