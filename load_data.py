import os
import random

import torch
import torch.utils.data as Data
import torchvision
from parameter import CHANNEL

def load_data(dataset_name, dataset_path, DOWNLOAD):
  if dataset_name == 'FashionMnist':
    # 载入anchor数据
    anchor_data = torchvision.datasets.FashionMNIST(
      root=dataset_path,
      train=True,
      transform=torchvision.transforms.ToTensor(),
      download=DOWNLOAD,
    )

    #载入加过旋转的数据,作为正例
    positive_data = torchvision.datasets.FashionMNIST(
      root=dataset_path,
      train=True,
      transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(180,fill=(0,)),
        torchvision.transforms.ToTensor()
      ]),
    )

    test_data = torchvision.datasets.FashionMNIST(
      root=dataset_path,
      train=False,
      transform=torchvision.transforms.ToTensor(),
    )
    num_train = anchor_data.targets.size().numel()

    num_test = test_data.targets.size().numel()

  elif dataset_name == 'CIFAR10':
    # 载入anchor数据
    anchor_data = torchvision.datasets.CIFAR10(
      root=dataset_path,
      train=True,
      transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(28),
        torchvision.transforms.ToTensor()
      ]),
      download=DOWNLOAD,
    )

    #载入加过旋转的数据,作为正例
    positive_data = torchvision.datasets.CIFAR10(
      root=dataset_path,
      train=True,
      transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(28),
        torchvision.transforms.RandomRotation(180,fill=(0)),
        torchvision.transforms.ToTensor()
      ]),
    )

    test_data = torchvision.datasets.CIFAR10(
      root=dataset_path,
      train=False,
      transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(28),
        torchvision.transforms.ToTensor()
      ]),
    )
    num_train = len(anchor_data.targets)

    num_test = len(test_data.targets)

  # 升维方便拼接，
  # TODO 可能有更好的方法
  train_data = torch.zeros(num_train, 3, CHANNEL, 28, 28)

  anchor_loader = Data.DataLoader(dataset=anchor_data, batch_size=num_train, shuffle=False)
  positive_loader = Data.DataLoader(dataset=positive_data, batch_size=num_train, shuffle=False)

  for ((anchor, _), (positive, _)) in zip(anchor_loader, positive_loader):
    for index, (anc, pos) in enumerate(zip(anchor, positive)):
      ng_index = random.randrange(0, num_train)
      while anchor_data.targets[ng_index] == anchor_data.targets[index]:
        ng_index = random.randrange(0, num_train)
      # We don't need label in this task.
      train_data[index][0] = anc
      train_data[index][1] = pos
      train_data[index][2] = anchor[ng_index]

  print('train_data.size : {}'.format(train_data.size()))


  return anchor_data, train_data, test_data, num_train, num_test

def load_data_no_triplet(dataset_name, dataset_path, DOWNLOAD):

  if dataset_name == 'FashionMnist':
    # 载入anchor数据
    train_data = torchvision.datasets.FashionMNIST(
      root=dataset_path,
      train=True,
      transform=torchvision.transforms.ToTensor(),
      download=DOWNLOAD,
    )

    test_data = torchvision.datasets.FashionMNIST(
      root=dataset_path,
      train=False,
      transform=torchvision.transforms.ToTensor(),
    )

    num_train = train_data.targets.size().numel()

    num_test = test_data.targets.size().numel()

  elif dataset_name == 'CIFAR10':
    # 载入anchor数据
    train_data = torchvision.datasets.CIFAR10(
      root=dataset_path,
      train=True,
      transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(28),
        torchvision.transforms.ToTensor(),
      ]),
      download=DOWNLOAD,
    )

    test_data = torchvision.datasets.CIFAR10(
      root=dataset_path,
      train=False,
      transform=torchvision.transforms.ToTensor(),
    )

    num_train = len(train_data.targets)
    num_test = len(test_data.targets)

  return train_data, test_data, num_train, num_test
