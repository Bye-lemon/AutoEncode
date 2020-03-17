import torch

a = torch.empty(50, 3, 28, 28)

print(a[:, 0, :, :].size())