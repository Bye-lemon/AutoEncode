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

class Encoder(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=128):
    super(Encoder, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    self.main = nn.Sequential(
      nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
      nn.ReLU(True),
      nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(self.dim_h * 2),
      nn.ReLU(True),
      nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(self.dim_h * 4),
      nn.ReLU(True),
      nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(self.dim_h * 8),
      nn.ReLU(True),
    )
    self.fc = nn.Sequential(
      nn.Linear(self.dim_h * (2 ** 3), self.n_z),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.main(x)
    x = x.squeeze()
    x = self.fc(x)
    return x

class Decoder(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=128):
    super(Decoder, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    self.proj = nn.Sequential(
      nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
      nn.ReLU()
    )

    self.main = nn.Sequential(
      nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
      nn.BatchNorm2d(self.dim_h * 4),
      nn.ReLU(True),
      nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
      nn.BatchNorm2d(self.dim_h * 2),
      nn.ReLU(True),
      nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.proj(x)
    x = x.view(-1, self.dim_h * 8, 7, 7)
    x = self.main(x)
    return x

encoder = Encoder(12)
decoder = Decoder(12)

encoder.load_state_dict(torch.load('./nets/encoder_params.pkl'))
decoder.load_state_dict(torch.load('./nets/decoder_params.pkl'))

N_TEST_IMG = 10
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
view_data = anchor_data.data[:N_TEST_IMG].view(10, 1, 28, 28).type(torch.FloatTensor)/255.
print(view_data.size())
for i in range(N_TEST_IMG):
  a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

code = encoder(view_data)
out = decoder(code)
out *= 255
for i in range(N_TEST_IMG):
  a[1][i].imshow(np.reshape(out.data[i].numpy(), (28, 28)), cmap='gray'); a[1][i].set_xticks(()); a[1][i].set_yticks(())

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
