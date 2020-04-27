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

EPOCH = 3
BATCH_SIZE = 100
LR = 0.005

MARGIN = 1
LAMBDA_T = 3
LAMBDA_U = 0
LAMBDA_V = 0
LAMBDA_Z = 0

DIMS = [64, 128, 192, 256]
# DIMS = [64, 128]
POSES = [1]
POSES.extend([i*10 for i in range(1, 5)])
POSES.extend([i*50 for i in range(1, 21)])
