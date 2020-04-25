import os

# DATASETNAME = 'FashionMnist'
# DATASETPATH = './fashion_mnist/'
# CHANNEL = 1
DATASETNAME = 'CIFAR10'
DATASETPATH = './cifar_10'
CHANNEL = 3
DOWNLOAD = False
if not(os.path.exists(DATASETPATH)) or not os.listdir(DATASETPATH):
  # nor dir or dir is empty
  DOWNLOAD = True

EPOCH = 4
BATCH_SIZE = 100
LR = 0.005

MARGIN = 1
LAMBDA_T = 3
LAMBDA_U = 1 / 100000
LAMBDA_V = 1
LAMBDA_Z = 1

DIMS = [64, 128, 192, 256]
POSES = [1]
POSES.extend([i*10 for i in range(1, 5)])
POSES.extend([i*50 for i in range(1, 21)])
