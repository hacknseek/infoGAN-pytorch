import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np
from model import *


# load model
model_path = './save_model/md-2-301.ckpt'
g = G().cuda()
g.load_state_dict(torch.load(model_path))

# hyperparam
sz_con = 54
sz_dis = 10
size_latent = sz_con + sz_dis

# input
input_label = 1
idx = np.tile(np.repeat([input_label], 10), 10) # [0,1,2,3,...]
one_hot = np.zeros((100, 10))
one_hot[range(100), idx] = 1
fix_noise = torch.Tensor(100, sz_con).uniform_(-1, 1)
z = torch.cat([torch.Tensor(one_hot), fix_noise], 1).view(-1, size_latent, 1, 1).cuda()
x_save = g(z)
save_image(x_save.data, './tmp/_demo.png', nrow=10)