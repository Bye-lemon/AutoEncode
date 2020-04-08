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

class ResDecoder(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=128):
    super(ResDecoder, self).__init__()

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

class DenseEncoder(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=128):
    super(DenseEncoder, self).__init__()

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

class DenseDecoder(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=128):
    super(DenseDecoder, self).__init__()

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
