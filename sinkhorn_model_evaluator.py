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

blur = 0.05
scaling = 0.95
backend = 'online'
lr = 0.045
device = 'cuda:3'
t = 100
dim = 2
batch_size = 1024
retrain = 50
model = torch.load('./models/8gaussian-moons_sinkhorn/lr0.045_t100_retrain50/cfm_v1_1.pt')
model.eval()
x0 = sample_8gaussians(1024)
xt = x0
for step in range(t):
    model_t = t / 100
    t_batch = torch.ones(batch_size).type_as(x0)
    vt = model(torch.cat([xt.to('cpu'), t_batch[:, None].to('cpu')], dim=-1))
    xt = xt + lr * vt
xt = xt.detach().numpy()
plt.figure(figsize=(6, 6))
plt.scatter(xt[:, 0], xt[:, 1], s=4, alpha=1, c="blue")
plt.xticks([])
plt.yticks([])
plt.savefig(f'./test/lr0.045_retrain50_t100/sink_dataext_retrain50_myevaluator')
