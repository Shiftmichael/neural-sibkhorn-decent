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

imgdir = './test/sinkhorn'
os.makedirs(imgdir, exist_ok=True)

blur = 0.05
scaling = 0.95
backend = 'online'
lr = 0.02
device = 'cuda:0'
t = 100
dim = 2
batch_size = 256
retrain = 50
FM = SinkhornFlowMatcher(blur = blur, scaling = scaling, backend = backend, lr = lr, device = device)

x0 = sample_8gaussians(batch_size).to(device)
x1 = sample_moons(batch_size).to(device)
traj = FM.sample_all_trajactory(x0, x1, t).to('cpu')
traj = traj.detach().numpy()
n = batch_size
plt.figure(figsize=(6, 6))
plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
plt.scatter(traj[t-1, :n, 0], traj[t-1, :n, 1], s=4, alpha=1, c="blue")
plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
plt.xticks([])
plt.yticks([])
plt.savefig(f'{imgdir}/sink')