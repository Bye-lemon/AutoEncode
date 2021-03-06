import torch.nn as nn
import torch

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
  """1x1 convolution"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
  expansion = 1
  __constants__ = ['downsample']

  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
              base_width=64, dilation=1, norm_layer=None):
    super(BasicBlock, self).__init__()
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 64:
        raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
        raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

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
      nn.ConvTranspose2d(self.dim_h * 2, self.n_channel, 4, stride=2),
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

    self._norm_layer = nn.BatchNorm2d
    self.dilation = 1
    self.inplanes = self.dim_h * 2
    self.groups = 1
    self.base_width = 64

    # 28 * 28 -> 7 * 7
    self.block1 = nn.Sequential(
      nn.Conv2d(self.n_channel, self.dim_h, kernel_size=4, stride=2, padding=1, bias=False),
      nn.ReLU(True),
      nn.Conv2d(self.dim_h, self.dim_h * 2, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(self.dim_h * 2),
      nn.ReLU(True),
    )
    # 7 * 7 -> 7 * 7
    self.layer1 = self._make_layer(BasicBlock, self.dim_h * 2, 3)
    self.relu = nn.ReLU(True)
    self.fc = nn.Sequential(
      nn.Linear(self.dim_h * 2 * 7 * 7, self.n_z),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.block1(x)
    out = self.layer1(x)
    out = out.view(-1, self.dim_h * 2 * 7 * 7)
    out = self.fc(out)
    return out 

  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes * block.expansion, stride),
          norm_layer(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes, groups=self.groups,
                          base_width=self.base_width, dilation=self.dilation,
                          norm_layer=norm_layer))

    return nn.Sequential(*layers)

class ResDecoder(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=128):
    super(ResDecoder, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    self._norm_layer = nn.BatchNorm2d
    self.dilation = 1
    self.inplanes = self.dim_h * 2
    self.groups = 1
    self.base_width = 64

    self.proj = nn.Sequential(
      nn.Linear(self.n_z, self.dim_h * 2 * 7 * 7),
      nn.ReLU()
    )

    self.layer1 = self._make_layer(BasicBlock, self.dim_h * 2, 3)

    self.main = nn.Sequential(
      nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
      nn.ConvTranspose2d(self.dim_h, self.dim_h, 4),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
      nn.ConvTranspose2d(self.dim_h, self.n_channel, 4, stride=2),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.proj(x)
    x = x.view(-1, self.dim_h * 2, 7, 7)
    x = self.layer1(x)
    x = self.main(x)
    return x

  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes * block.expansion, stride),
          norm_layer(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes, groups=self.groups,
                          base_width=self.base_width, dilation=self.dilation,
                          norm_layer=norm_layer))

    return nn.Sequential(*layers)
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
    features = torch.cat((features, new_feature), 1)
    new_feature = self.layer1(features)
    features = torch.cat((features, new_feature), 1)
    out = self.layer2(features)
    out = out.view(-1, self.dim_h * 7 * 7)
    out = self.fc(out)
    return out 

class DenseDecoder(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=128):
    super(DenseDecoder, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    self.proj = nn.Sequential(
      nn.Linear(self.n_z, self.dim_h * 7 * 7),
      nn.ReLU()
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
    self.main = nn.Sequential(
      nn.ConvTranspose2d(self.dim_h, self.dim_h, 4),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
      nn.ConvTranspose2d(self.dim_h, self.dim_h, 4),
      nn.BatchNorm2d(self.dim_h),
      nn.ReLU(True),
      nn.ConvTranspose2d(self.dim_h, self.n_channel, 4, stride=2),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.proj(x)
    x = x.view(-1, self.dim_h, 7, 7)
    features = x
    new_feature = self.layer0(features)
    features = torch.cat((features, new_feature), 1)
    new_feature = self.layer1(features)
    features = torch.cat((features, new_feature), 1)
    out = self.layer2(features)
    out = self.main(out)
    return out 

class FcEncoder(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=128):
    super(FcEncoder, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    self.main = nn.Sequential(
      nn.Linear(28*28*self.n_channel, self.dim_h * 8),
      nn.Tanh(),
      nn.Linear(self.dim_h * 8, self.dim_h * 4),
      nn.Tanh(),
      nn.Linear(self.dim_h * 4, self.dim_h * 2),
      nn.Tanh(),
      nn.Linear(self.dim_h * 2, self.n_z),
      nn.Sigmoid(),
    )

  def forward(self, x):
    x = x.view(-1, 28 * 28 * self.n_channel)
    x = self.main(x)
    return x

class FcDecoder(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=128):
    super(FcDecoder, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    self.main = nn.Sequential(
      nn.Linear(self.n_z, self.dim_h * 2),
      nn.Tanh(),
      nn.Linear(self.dim_h * 2, self.dim_h * 4),
      nn.Tanh(),
      nn.Linear(self.dim_h * 4, self.dim_h * 8),
      nn.Tanh(),
      nn.Linear(self.dim_h * 8, 28*28*self.n_channel),
      nn.Sigmoid(),
    )
  def forward(self, x):
    x = self.main(x)
    x = x.view(-1, self.n_channel, 28, 28)
    return x

class Encoder1(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=64):
    super(Encoder1, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    self.main = nn.Sequential(
      nn.Conv2d(self.n_channel, self.dim_h, 5, 1, 2, bias=False),
      nn.ReLU(True),
      nn.BatchNorm2d(self.dim_h),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(self.dim_h, self.dim_h, 5, 1, 2, bias=False),
      nn.ReLU(True),
      nn.BatchNorm2d(self.dim_h),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(self.dim_h, self.dim_h, 5, 1, 2, bias=False),
      nn.ReLU(True),
      nn.BatchNorm2d(self.dim_h),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(self.dim_h, self.dim_h * 2, 5, 1, 2, bias=False),
      nn.ReLU(True),
      nn.BatchNorm2d(self.dim_h * 2),
      nn.MaxPool2d(2, 2),
    )
    self.fc = nn.Sequential(
      nn.Linear(self.dim_h * 2, self.n_z),
      nn.Sigmoid(),
    )

  def forward(self, x):
    x = self.main(x)
    x = x.squeeze()
    x = self.fc(x)
    return x

class Decoder1(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=128):
    super(Decoder1, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    self.proj = nn.Sequential(
      nn.Linear(self.n_z, self.dim_h * 2),
      nn.ReLU()
    )

    self.main = nn.Sequential(
      nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 3),
      nn.ReLU(True),
      nn.BatchNorm2d(self.dim_h),
      nn.ConvTranspose2d(self.dim_h, self.dim_h, 3, stride=2),
      nn.ReLU(True),
      nn.BatchNorm2d(self.dim_h),
      nn.ConvTranspose2d(self.dim_h, self.dim_h, 2, stride=2),
      nn.ReLU(True),
      nn.BatchNorm2d(self.dim_h),
      nn.ConvTranspose2d(self.dim_h, self.n_channel, 2, stride=2),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.proj(x)
    x = x.view(-1, self.dim_h * 2, 1, 1)
    x = self.main(x)
    return x

class ResEncoder1(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=16):
    super(ResEncoder1, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    self._norm_layer = nn.BatchNorm2d
    self.dilation = 1
    self.inplanes = self.dim_h * 2
    self.groups = 1
    self.base_width = 64

    # 28 * 28 -> 28 * 28
    self.block1 = nn.Sequential(
      nn.Conv2d(self.n_channel, self.dim_h, kernel_size=3, stride=1, padding=1, bias=False),
      nn.ReLU(True),
      nn.BatchNorm2d(self.dim_h),
    )
    # 28 * 28 -> 28 * 28
    self.layer1 = self._make_layer(BasicBlock, self.dim_h, self.dim_h, 3)
    self.layer2 = self._make_layer(BasicBlock, self.dim_h, self.dim_h * 2, 1)
    self.layer3 = self._make_layer(BasicBlock, self.dim_h * 2, self.dim_h * 2, 3)
    self.layer4 = self._make_layer(BasicBlock, self.dim_h * 2, self.dim_h * 4, 1)
    self.layer5 = self._make_layer(BasicBlock, self.dim_h * 4, self.dim_h * 4, 3)
    self.relu = nn.ReLU(True)
    self.bn = nn.BatchNorm2d(self.dim_h * 4)
    self.fc = nn.Sequential(
      nn.Linear(self.dim_h * 4 * 28 * 28, self.n_z),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.block1(x)
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = out.view(-1, self.dim_h * 4 * 28 * 28)
    out = self.fc(out)
    return out 

  def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv1x1(inplanes, planes * block.expansion, stride),
          norm_layer(planes * block.expansion),
      )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(inplanes, planes, groups=self.groups,
                          base_width=self.base_width, dilation=self.dilation,
                          norm_layer=norm_layer))

    return nn.Sequential(*layers)

class ResDecoder1(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=16):
    super(ResDecoder1, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    self._norm_layer = nn.BatchNorm2d
    self.dilation = 1
    self.groups = 1
    self.base_width = 64

    self.proj = nn.Sequential(
      nn.Linear(self.n_z, self.dim_h * 4 * 28 * 28),
      nn.ReLU()
    )

    self.layer1 = self._make_layer(BasicBlock, self.dim_h * 4, self.dim_h * 4, 3)
    self.layer2 = self._make_layer(BasicBlock, self.dim_h * 4, self.dim_h * 2, 1)
    self.layer3 = self._make_layer(BasicBlock, self.dim_h * 2, self.dim_h * 2, 3)
    self.layer4 = self._make_layer(BasicBlock, self.dim_h * 2, self.dim_h, 1)
    self.layer5 = self._make_layer(BasicBlock, self.dim_h, self.dim_h, 3)
    self.relu = nn.ReLU(True)
    self.bn = nn.BatchNorm2d(self.dim_h * 4)

    self.main = nn.Sequential(
      nn.Conv2d(self.dim_h, self.n_channel, kernel_size=3, stride=1, padding=1, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.proj(x)
    x = x.view(-1, self.dim_h * 4, 28, 28)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.main(x)
    return x

  def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv1x1(inplanes, planes * block.expansion, stride),
          norm_layer(planes * block.expansion),
      )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(inplanes, planes, groups=self.groups,
                          base_width=self.base_width, dilation=self.dilation,
                          norm_layer=norm_layer))

    return nn.Sequential(*layers)

class Encoder2(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=64):
    super(Encoder2, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    self.main = nn.Sequential(
      nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1, bias=False),
      nn.ReLU(True),
      nn.BatchNorm2d(self.n_channel),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1, bias=False),
      nn.ReLU(True),
      nn.BatchNorm2d(self.n_channel),
      nn.MaxPool2d(2, 2),
    )
    self.fc = nn.Sequential(
      nn.Linear(self.n_channel * 7 * 7, 192),
      nn.ReLU(True),
      nn.Linear(192, 135),
      nn.ReLU(True),
      nn.Linear(135, 78),
      nn.ReLU(True),
      nn.Linear(78, self.n_z),
      nn.Sigmoid(),
    )

  def forward(self, x):
    x = self.main(x)
    x = x.view(-1, self.n_channel * 7 * 7)
    # x = x.squeeze()
    x = self.fc(x)
    return x

class Decoder2(nn.Module):
  def __init__(self, n_z, n_channel=1, dim_h=128):
    super(Decoder2, self).__init__()

    self.n_channel = n_channel
    self.dim_h = dim_h
    self.n_z = n_z

    self.proj = nn.Sequential(
      nn.Linear(self.n_z, 78),
      nn.ReLU(),
      nn.Linear(78, 135),
      nn.ReLU(),
      nn.Linear(135, 192),
      nn.ReLU(),
      nn.Linear(192, self.n_channel * 7 * 7),
      nn.ReLU(),
    )

    self.main = nn.Sequential(
      nn.MaxUnpool2d(2, stride=2),
      nn.ConvTranspose2d(self.n_channel, self.n_channel, 1),
      nn.ReLU(True),
      nn.BatchNorm2d(self.n_channel),
      nn.MaxUnpool2d(2, stride=2),
      nn.ConvTranspose2d(self.n_channel, self.n_channel, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.proj(x)
    x = x.view(-1, self.n_channel, 7, 7)
    x = self.main(x)
    return x
