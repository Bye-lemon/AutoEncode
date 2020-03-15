import os
import random

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
      nn.Linear(28 * 28, 128),
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
      nn.Linear(128, 28 * 28),
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

trainY = torch.empty(num_train, TARGET_DIM)
trainLabel = anchor_data.targets.numpy()

testY = torch.empty(num_test, TARGET_DIM)
testLable = test_data.targets.numpy()

for index, (x, label) in enumerate(anchor_data):
  x = x.view(-1, 28 * 28)
  encoded = autocoder.encoder(x)
  trainY[index] = encoded

for index, (x, label) in enumerate(test_data):
  x = x.view(-1, 28 * 28)
  encoded = autocoder.encoder(x)
  testY[index] = encoded

trainY = trainY.detach().numpy()
testY = testY.detach().numpy()

trainY = compactBit(trainY)
testY = compactBit(testY)

ntrain = trainY.shape[0]
ntest = testY.shape[0]
Wture = np.zeros((ntrain, ntest))
for i in range(ntrain):
    for j in range(ntest):
        if trainLabel[i] == testLable[j]:
            Wture[i, j] = 1

hamdis = hamming_dist(trainY, testY)

for i in range(ntest):
    index_ = hamdis[:, i].argsort()
    Wture[:, i] = Wture[index_, i]

pos = 10
retrieved_good_pairs = Wture[:pos, :].sum()
row, col = Wture[:pos, :].shape
total_pairs = row * col
precision = retrieved_good_pairs / (total_pairs + 1e-3)

print('precision: {}'.format(precision))