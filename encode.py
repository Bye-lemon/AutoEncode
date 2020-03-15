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
#载入加过旋转的数据,作为正例
positive_data = torchvision.datasets.FashionMNIST(
  root=DATASETPATH,
  train=True,
  transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(180, fill=(0,)),
    torchvision.transforms.ToTensor()
  ]),
)
# 训练集总数
num_train = anchor_data.targets.size().numel()

# 升维方便拼接，
# TODO 可能有更好的方法
print(anchor_data[0][0].size())
train_data = torch.zeros(num_train, 3, anchor_data[0][0].size(1), anchor_data[0][0].size(2))

for index, anchor in enumerate(anchor_data):
  ng_index = random.randrange(0, num_train)
  while anchor_data.targets[ng_index] == anchor_data.targets[index]:
    ng_index = random.randrange(0, num_train)
  # We don't need label in this task.
  train_data[index][0] = anchor[0]
  train_data[index][1] = positive_data[index][0]
  train_data[index][2] = anchor_data[ng_index][0]


print('train_data.size : {}'.format(train_data.size()))

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

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

autocoder = AutoEncoder()

optimizer = torch.optim.Adam(autocoder.parameters(), lr=LR)
mseLoss_fun = nn.MSELoss()
tripletLoss_fun = nn.TripletMarginLoss(margin=MARGIN, p=1)

for epoch in range(EPOCH):
  for step, x in enumerate(train_loader):
    b_x = x.view(BATCH_SIZE, 3, 28 * 28)
    b_y = x.view(BATCH_SIZE, 3, 28 * 28)

    (anc_encoded, anc_decoded, pos_encoded,
     pos_decoded, neg_encoded, neg_decoded) = autocoder(b_x)

    encode_loss = mseLoss_fun(anc_decoded, b_y[:, 0, :]) + \
                  mseLoss_fun(pos_decoded, b_y[:, 1, :]) + \
                  mseLoss_fun(neg_decoded, b_y[:, 2, :])
    triplet_loss = tripletLoss_fun(anc_encoded, pos_encoded, neg_encoded)

    loss = encode_loss + LAMBDA_T * triplet_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
      print('Epoch: {}, train_loss: {}'.format(epoch, loss.data.numpy()))

torch.save(autocoder, 'autoencoder.pkl')

trainY = torch.empty(num_train, TARGET_DIM)
trainLabel = anchor_data.targets.numpy()

testY = torch.empty(num_test, TARGET_DIM)
testLable = test_data.targets.numpy()

for index, (x, label) in enumerate(anchor_data):
  encoded, _ = autocoder(x)
  trainY[index] = encoded

for index, (x, label) in enumerate(test_data):
  encoded, _ = autocoder(x)
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






