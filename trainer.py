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
import math
import logging

class log_gaussian:

  def __call__(self, x, mu, var):

    logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)
    
    return logli.sum(1).mean().mul(-1)

class Trainer:

  def __init__(self, G, FE, D, Q):

    self.G = G
    self.FE = FE
    self.D = D
    self.Q = Q

    # hyperparameters
    '''
    Batch size: 64 
    Learning rate for the discriminator: 2e-4 
    Learning rate for the generator and Q: 1e-3 
    c_size (size of meaningful codes) = 10 
    Total epochs = 80 
    Optimizer: Adam 
    '''
    self.num_epoch = 80
    self.batch_size = 64
    self.lr_D = 2e-4
    self.lr_GQ = 1e-3
    self.num_labels = 10

    # latent parameters
    self.sz_con = 54
    self.sz_dis = 10
    self.size_latent = self.sz_con + self.sz_dis

    # logging
    self.log_iter = 100
    logging.basicConfig(filename='example.log',level=logging.DEBUG, filemode='w')


  # def _noise_sample(self, dis_c, con_c, noise, bs):
  def _noise_sample(self, dis_c, con_c, bs):

    idx = np.random.randint(self.num_labels, size=bs)
    c = np.zeros((bs, self.num_labels))
    c[range(bs),idx] = 1.0

    dis_c.data.copy_(torch.Tensor(c))
    con_c.data.uniform_(-1.0, 1.0)
    # noise.data.uniform_(-1.0, 1.0)
    # z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
    z = torch.cat([dis_c, con_c], 1).view(-1, self.size_latent, 1, 1)

    return z, idx

  def train(self):

    real_x = torch.FloatTensor(self.batch_size, 1, 28, 28).cuda()
    label = torch.FloatTensor(self.batch_size).cuda()
    dis_c = torch.FloatTensor(self.batch_size, self.sz_dis).cuda()
    con_c = torch.FloatTensor(self.batch_size, self.sz_con).cuda()
    # noise = torch.FloatTensor(self.batch_size, 62).cuda()

    real_x = Variable(real_x)
    label = Variable(label, requires_grad=False)
    dis_c = Variable(dis_c)
    con_c = Variable(con_c)
    # noise = Variable(noise)

    criterionD = nn.BCELoss().cuda()
    criterionQ_dis = nn.CrossEntropyLoss().cuda()
    criterionQ_con = log_gaussian()

    optimD = optim.Adam([{'params':self.FE.parameters()}, {'params':self.D.parameters()}], lr=self.lr_D , betas=(0.5, 0.99))
    optimG = optim.Adam([{'params':self.G.parameters()}, {'params':self.Q.parameters()}], lr=self.lr_GQ , betas=(0.5, 0.99))

    dataset = dset.MNIST('./dataset', transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

    # fixed random variables
    repeat_num = math.ceil(self.batch_size/self.num_labels)
    repeat_num = 10

    # c = np.linspace(-1, 1, self.num_labels).reshape(1, -1)
    # c = np.repeat(c, repeat_num, 0).reshape(-1, 1)
    # c = c[:self.batch_size]

    # c1 = np.hstack([c, np.zeros_like(c)])
    # c2 = np.hstack([np.zeros_like(c), c])

    # idx = np.arange(10).repeat(10) # [0,0,0,0,0..]
    idx = np.tile(np.arange(10), 10) # [0,1,2,3,...]
    # idx = idx[:self.batch_size] # cut down excessive idx
    one_hot = np.zeros((100, 10))
    one_hot[range(100), idx] = 1
    fix_noise = torch.Tensor(10, 1 ,self.sz_con).uniform_(-1, 1).repeat(1, 10, 1)
    fix_noise = fix_noise.view(100, self.sz_con)
    # print(fix_noise.size()), exit()

    for epoch in range(self.num_epoch):
      for num_iters, batch_data in enumerate(dataloader, 0):

        # real part
        optimD.zero_grad()
        
        x, _ = batch_data # x: torch.Size([64, 1, 28, 28])

        bs = x.size(0)
        real_x.data.resize_(x.size())
        label.data.resize_(bs)
        dis_c.data.resize_(bs, self.sz_dis)
        con_c.data.resize_(bs, self.sz_con)
        # noise.data.resize_(bs, 62)
        
        real_x.data.copy_(x)
        fe_out1 = self.FE(real_x) # feout: torch.Size([64, 1024, 1, 1])
        probs_real = self.D(fe_out1)
        label.data.fill_(1)
        loss_real = criterionD(probs_real, label)
        loss_real.backward()

        # fake part
        z, idx = self._noise_sample(dis_c, con_c, bs) # z: torch.Size([64, 74, 1, 1]) (74=old latent_size)
        fake_x = self.G(z) # fake_x: torch.Size([64, 1, 28, 28])
        fe_out2 = self.FE(fake_x.detach())
        probs_fake = self.D(fe_out2)
        label.data.fill_(0)
        loss_fake = criterionD(probs_fake, label)
        loss_fake.backward()

        D_loss = loss_real + loss_fake

        optimD.step()
        
        # G and Q part
        optimG.zero_grad()

        fe_out = self.FE(fake_x)
        probs_fake = self.D(fe_out)
        label.data.fill_(1.0)

        reconstruct_loss = criterionD(probs_fake, label)
        
        q_logits, q_mu, q_var = self.Q(fe_out)
        class_ = torch.LongTensor(idx).cuda()
        target = Variable(class_)
        dis_loss = criterionQ_dis(q_logits, target)
        # con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1
        
        # G_loss = reconstruct_loss + dis_loss + con_loss
        G_loss = reconstruct_loss + dis_loss
        G_loss.backward()
        optimG.step()

        if num_iters % self.log_iter == 0:

          msg = 'Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
            epoch, num_iters, D_loss.data.cpu().numpy(),
            G_loss.data.cpu().numpy())
          
          print(msg)
          logging.info(msg)

          # noise.data.copy_(fix_noise)
          # --dis_c.data.copy_(torch.Tensor(one_hot))

          # --con_c.data.copy_(fix_noise)
          # z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
          # z = torch.cat([dis_c, con_c], 1).view(-1, 64, 1, 1)

          z = torch.cat([torch.Tensor(one_hot), fix_noise], 1).view(-1, self.size_latent, 1, 1).cuda()
          x_save = self.G(z)
          save_image(x_save.data, './tmp/c1_{}.png'.format(epoch+1), nrow=10)

          # con_c.data.copy_(fix_noise)
          # # z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
          # z = torch.cat([dis_c, con_c], 1).view(-1, 64, 1, 1)
          # x_save = self.G(z)
          # save_image(x_save.data, './tmp/c2.png', nrow=10)

          # save model
          torch.save(self.G.state_dict(), './save_model/md-{}.ckpt'.format(epoch+1))
