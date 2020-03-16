import os
import random
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

from hashing_utils import *

DATASETPATH = './fashion_mnist/'
DOWNLOAD = False

EPOCH = 10
BATCH_SIZE = 50
LR = 0.005

TARGET_DIM = 8
MARGIN = 1
LAMBDA_T = 3

if not(os.path.exists(DATASETPATH)) or not os.listdir(DATASETPATH):
  # nor dir or dir is empty
  DOWNLOAD = True

# 载入anchor数据
anchor_data = torchvision.datasets.FashionMNIST(
  root=DATASETPATH,
  train=True,
  transform=torchvision.transforms.ToTensor(),
  download=DOWNLOAD,
)

num_train = anchor_data.targets.size().numel()

test_data = torchvision.datasets.FashionMNIST(
  root=DATASETPATH,
  train=False,
  transform=torchvision.transforms.ToTensor(),
)

num_test = test_data.targets.size().numel()

class AutoEncoder(nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()

    self.encoder = nn.Sequential(
      nn.Linear(28 * 28, 392),
      nn.Tanh(),
      nn.Linear(392, 196),
      nn.Tanh(),
      nn.Linear(196, 128),
      nn.Tanh(),
      nn.Linear(128, 64),
      nn.Tanh(),
      nn.Linear(64, TARGET_DIM),
      nn.Sigmoid(),
    )
    self.decoder = nn.Sequential(
      nn.Linear(TARGET_DIM, 64),
      nn.Tanh(),
      nn.Linear(64, 128),
      nn.Tanh(),
      nn.Linear(128, 196),
      nn.Tanh(),
      nn.Linear(196, 392),
      nn.Tanh(),
      nn.Linear(392, 28 * 28),
      nn.Sigmoid(),
    )

  def forward(self, x):
    anc_encode = self.encoder(x[:, 0, :])
    anc_decode = self.decoder(anc_encode)
    pos_encode = self.encoder(x[:, 1, :])
    pos_decode = self.decoder(pos_encode)
    neg_encode = self.encoder(x[:, 2, :])
    neg_decode = self.decoder(neg_encode)
    return anc_encode, anc_decode, pos_encode, pos_decode, neg_encode, neg_decode


autocoder = torch.load('autoencoder.pkl')

N_TEST_IMG = 10
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
view_data = anchor_data.data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
for i in range(N_TEST_IMG):
  a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

for i in range(N_TEST_IMG):
  code = autocoder.encoder(view_data[i])
  out = autocoder.decoder(code)
  out *= 255
  a[1][i].imshow(np.reshape(out.data.numpy(), (28, 28)), cmap='gray'); a[1][i].set_xticks(()); a[1][i].set_yticks(())

plt.show()
input()

trainY = torch.empty(num_train, TARGET_DIM)
trainLabel = anchor_data.targets.numpy()

print('after mking empty trainY')

testY = torch.empty(num_test, TARGET_DIM)
testLable = test_data.targets.numpy()

print('after mking empty testY')

for index, (x, label) in enumerate(anchor_data):
  x = x.view(-1, 28 * 28)
  encoded = autocoder.encoder(x)
  trainY[index] = encoded

print('after mking trainY')

for index, (x, label) in enumerate(test_data):
  x = x.view(-1, 28 * 28)
  encoded = autocoder.encoder(x)
  testY[index] = encoded

print('after mking testY')

trainY = trainY.detach().numpy()
testY = testY.detach().numpy()

trainY = compactBit(trainY)
testY = compactBit(testY)

print('after compact')

ntrain = trainY.shape[0]
ntest = testY.shape[0]
Wture = np.zeros((ntrain, ntest))

print('calculating Wtrue')

for i in tqdm(range(ntrain)):
    for j in range(ntest):
        if trainLabel[i] == testLable[j]:
            Wture[i, j] = 1

print('calculating hamming dist')
hamdis = hamming_dist(trainY, testY)

print('finding nearest neighbors')
for i in tqdm(range(ntest)):
    index_ = hamdis[:, i].argsort()
    Wture[:, i] = Wture[index_, i]

print('calculating precision')
pos = 3
retrieved_good_pairs = Wture[:pos, :].sum()
row, col = Wture[:pos, :].shape
total_pairs = row * col
precision = retrieved_good_pairs / (total_pairs + 1e-3)

print('precision: {}'.format(precision))