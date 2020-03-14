import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image


DATASETPATH = './fashion_mnist/'
DOWNLOAD = False

# anchor_data = torchvision.datasets.FashionMNIST(
#   root=DATASETPATH,
#   train=True,
#   transform=torchvision.transforms.ToTensor(),
#   download=DOWNLOAD,
# )
#载入加过旋转的数据,作为正例
positive_data = torchvision.datasets.FashionMNIST(
  root=DATASETPATH,
  train=True,
  # transform=torchvision.transforms.Compose([
  #   torchvision.transforms.RandomRotation(degrees=180),
  #   torchvision.transforms.ToTensor()
  # ]),
  # transform=torchvision.transforms.RandomRotation(degrees=180),
  download=False,
)
plt.imshow(positive_data.data[2].numpy())
plt.show()
a = positive_data.data[2]
print(type(a))
print(a.size())
a = TF.to_pil_image(positive_data.data[2].numpy())
plt.imshow(a)
plt.show()
print(type(a))
a = a.rotate(45)
plt.imshow(a)
plt.show()
# print(torch.equal(anchor_data.data, positive_data.data))