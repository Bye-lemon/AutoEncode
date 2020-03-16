from parameter import DATASETPATH

def load_data(dataset_path):

  if not(os.path.exists(dataset_path)) or not os.listdir(dataset_path):
    # nor dir or dir is empty
    DOWNLOAD = True

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
      torchvision.transforms.RandomRotation(180, fill=(0,)),
      torchvision.transforms.ToTensor()
    ]),
  )

  # 训练集总数
  num_train = anchor_data.targets.size().numel()

  # 升维方便拼接，
  # TODO 可能有更好的方法
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

  test_data = torchvision.datasets.FashionMNIST(
    root=dataset_path,
    train=False,
    transform=torchvision.transforms.ToTensor(),
  )

  num_test = test_data.targets.size().numel()

  return anchor_data, train_data, test_data, num_train, num_test