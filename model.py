import torch.nn as nn

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

class ResEncoder(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=128):
    super(ResEncoder, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    # 28 * 28 -> 7 * 7
    self.block1 = nn.Sequential(
      nn.Conv2d(self.n_channel, self.dim_h, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
      nn.Conv2d(self.dim_h, self.dim_h, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
    )
    # 7 * 7 -> 7 * 7
    self.block2 = nn.Sequential(
      nn.Conv2d(self.dim_h, self.dim_h, 3, 1, 1, bias=False),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
      nn.Conv2d(self.dim_h, self.dim_h, 3, 1, 1, bias=False),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
    )
    self.fc = nn.Sequential(
      nn.Linear(self.dim_h * 7 * 7, self.n_z),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.block1(x)
    identity = x
    x = self.block2(x)
    x += identity
    x = x.view(-1, self.dim_h * 7 * 7)
    x = self.fc(x)
    return x


class DenseEncoder(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=128):
    super(DenseEncoder, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    self.block1 = nn.Sequential(
      nn.Conv2d(self.n_channel, self.dim_h, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
      nn.Conv2d(self.dim_h, self.dim_h, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
    )
    self.layer0 = nn.Sequential(
      nn.Conv2d(self.dim_h, self.dim_h, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
      nn.Conv2d(self.dim_h, self.dim_h, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
    )
    self.layer1 = nn.Sequential(
      nn.Conv2d(self.dim_h * 2, self.dim_h, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
      nn.Conv2d(self.dim_h, self.dim_h, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
    )
    self.layer2 = nn.Sequential(
      nn.Conv2d(self.dim_h * 3, self.dim_h, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
      nn.Conv2d(self.dim_h, self.dim_h, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
    )
    self.fc = nn.Sequential(
      nn.Linear(self.dim_h * 7 * 7, self.n_z),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.block1(x)
    features = x
    new_feature = self.layer0(features)
    features.append(new_feature)
    new_feature = self.layer1(features)
    features.append(new_feature)
    x = self.layer2(features)
    x = x.view(-1, self.dim_h * 7 * 7)
    x = self.fc(x)
    return x
