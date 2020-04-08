import os

DATASETNAME = 'FashionMnist'
DATASETPATH = './fashion_mnist/'
DOWNLOAD = False
if not(os.path.exists(DATASETPATH)) or not os.listdir(DATASETPATH):
  # nor dir or dir is empty
  DOWNLOAD = True

EPOCH = 1
BATCH_SIZE = 50
LR = 0.005

MARGIN = 1
LAMBDA_T = 3
LAMBDA_U = 1 / 25000
LAMBDA_V = 3
LAMBDA_Z = 1 / 500

DIMS = [12]
POSES = [1]
POSES.extend([i*10 for i in range(1, 5)])
POSES.extend([i*50 for i in range(1, 21)])
