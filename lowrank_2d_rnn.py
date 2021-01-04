"""
The MIT License (MIT)

Copyright (c) 2020 Zongyi Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

activation = F.relu

################################################################
# lowrank layers
################################################################
class LowRank2d(nn.Module):
    def __init__(self, in_channels, out_channels, s, ker_width, rank):
        super(LowRank2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = s
        self.n = s*s
        self.rank = rank

        self.phi = DenseNet([in_channels, ker_width, in_channels*out_channels*rank], torch.nn.ReLU)
        self.psi = DenseNet([in_channels, ker_width, in_channels*out_channels*rank], torch.nn.ReLU)


    def forward(self, v):
        batch_size = v.shape[0]

        phi_eval = self.phi(v).reshape(batch_size, self.n, self.out_channels, self.in_channels, self.rank)
        psi_eval = self.psi(v).reshape(batch_size, self.n, self.out_channels, self.in_channels, self.rank)

        # print(psi_eval.shape, v.shape, phi_eval.shape)
        v = torch.einsum('bnoir,bni,bmoir->bmo',psi_eval, v, phi_eval)

        return v



class MyNet(torch.nn.Module):
    def __init__(self, s, width=16, ker_width=256, rank=16):
        super(MyNet, self).__init__()
        self.s = s
        self.width = width
        self.ker_width = ker_width
        self.rank = rank

        self.fc0 = nn.Linear(12, self.width)

        self.conv0 = LowRank2d(width, width, s, ker_width, rank)
        self.conv1 = LowRank2d(width, width, s, ker_width, rank)
        self.conv2 = LowRank2d(width, width, s, ker_width, rank)
        self.conv3 = LowRank2d(width, width, s, ker_width, rank)

        self.w0 = nn.Linear(self.width, self.width)
        self.w1 = nn.Linear(self.width, self.width)
        self.w2 = nn.Linear(self.width, self.width)
        self.w3 = nn.Linear(self.width, self.width)
        self.bn0 = torch.nn.BatchNorm1d(self.width)
        self.bn1 = torch.nn.BatchNorm1d(self.width)
        self.bn2 = torch.nn.BatchNorm1d(self.width)
        self.bn3 = torch.nn.BatchNorm1d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        batch_size = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]
        x = x.view(batch_size, size_x*size_y, -1)

        x = self.fc0(x)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.bn0(x.reshape(-1, self.width)).view(batch_size, size_x*size_y, self.width)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.bn1(x.reshape(-1, self.width)).view(batch_size, size_x*size_y, self.width)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.bn2(x.reshape(-1, self.width)).view(batch_size, size_x*size_y, self.width)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.bn3(x.reshape(-1, self.width)).view(batch_size, size_x*size_y, self.width)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.view(batch_size, size_x, size_y, -1)
        return x

class Net2d(nn.Module):
    def __init__(self, width=12, ker_width=128, rank=4):
        super(Net2d, self).__init__()

        self.conv1 = MyNet(s=64, width=width, ker_width=ker_width, rank=rank)


    def forward(self, x):
        x = self.conv1(x)
        return x


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

################################################################
# configs
################################################################
# TRAIN_PATH = 'data/ns_data_V10000_N1200_T20.mat'
# TEST_PATH = 'data/ns_data_V10000_N1200_T20.mat'
# TRAIN_PATH = 'data/ns_data_V1000_N1000_train.mat'
# TEST_PATH = 'data/ns_data_V1000_N1000_train_2.mat'
# TRAIN_PATH = 'data/ns_data_V1000_N5000.mat'
# TEST_PATH = 'data/ns_data_V1000_N5000.mat'
TRAIN_PATH = 'data/ns_data_V100_N1000_T50_1.mat'
TEST_PATH = 'data/ns_data_V100_N1000_T50_2.mat'

ntrain = 1000
ntest = 200

batch_size = 5
batch_size2 = batch_size

epochs = 500
learning_rate = 0.0025
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

path = 'ns_lowrank_rnn_V100_T40_N'+str(ntrain)+'_ep' + str(epochs) + '_m'
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path


runtime = np.zeros(2, )
t1 = default_timer()


sub = 1
S = 64
T_in = 10
T = 40
step = 1


################################################################
# load dataset
################################################################
reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])


train_a = train_a.reshape(ntrain,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)

gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])

train_a = torch.cat((gridx.repeat([ntrain,1,1,1]), gridy.repeat([ntrain,1,1,1]), train_a), dim=-1)
test_a = torch.cat((gridx.repeat([ntest,1,1,1]), gridy.repeat([ntest,1,1,1]), test_a), dim=-1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

################################################################
# training and evaluation
################################################################
model = Net2d().cuda()
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

print(model.count_params())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


myloss = LpLoss(size_average=False)
gridx = gridx.to(device)
gridy = gridy.to(device)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:-2], im,
                            gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        # l2_full.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:-2], im,
                                gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)


            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
          test_l2_full / ntest)
# torch.save(model, path_model)


# pred = torch.zeros(test_u.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# with torch.no_grad():
#     for x, y in test_loader:
#         test_l2 = 0;
#         x, y = x.cuda(), y.cuda()
#
#         out = model(x)
#         out = y_normalizer.decode(out)
#         pred[index] = out
#
#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         print(index, test_l2)
#         index = index + 1

# scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})

