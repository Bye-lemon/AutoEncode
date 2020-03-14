import os
import random

import torch
import torchvision
import matplotlib.pyplot as plt

DATASETPATH = './fashion_mnist/'
DOWNLOAD = False

if not(os.path.exists(DATASETPATH)) or not os.listdir(DATASETPATH):
  # nor dir or dir is empty
  DOWNLOAD = True

def generate_train_data():
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
      torchvision.transforms.RandomRotation((30, 60)),
      torchvision.transforms.ToTensor()
    ]),
    download=False,
  )
  # 训练集总数
  num_train = anchor_data.targets.size().numel()
  print(torch.equal(anchor_data.data, positive_data.data))
  # 升维方便拼接，
  # TODO 可能有更好的方法
  anchor_data.data = torch.unsqueeze(anchor_data.data, dim=1)
  positive_data.data = torch.unsqueeze(positive_data.data, dim=1)
  anchor_data.data = torch.unsqueeze(anchor_data.data, dim=1)
  positive_data.data = torch.unsqueeze(positive_data.data, dim=1)
  pairs = []

  for index, anchor in enumerate(anchor_data.data):
    ng_index = random.randrange(0, num_train)
    while anchor_data.targets[ng_index] == anchor_data.targets[index]:
      ng_index = random.randrange(0, num_train)
    pairs.append(torch.cat([anchor, positive_data.data[index], anchor_data.data[ng_index]], dim=1))
  return torch.cat(pairs, dim=0)

train_data = generate_train_data()
print(train_data.size())
sum = 0
for i in range(100): 
  if torch.equal(train_data[i][0], train_data[i][1]):
    sum += 1

print(sum)

# plt.imshow(train_data[1][0].numpy(), cmap='gray')
# plt.show()
# plt.imshow(train_data[1][1].numpy(), cmap='gray')
# plt.show()
# plt.imshow(train_data[1][2].numpy(), cmap='gray')
# plt.show()