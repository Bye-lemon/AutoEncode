import os

DATASETNAME = 'FashionMnist'
DATASETPATH = './fashion_mnist/'
DOWNLOAD = False
if not(os.path.exists(DATASETPATH)) or not os.listdir(DATASETPATH):
  # nor dir or dir is empty
  DOWNLOAD = True

EPOCH = 3
BATCH_SIZE = 100
LR = 0.005

MARGIN = 1
LAMBDA_T = 3
LAMBDA_U = 1 / 500
LAMBDA_V = 10
LAMBDA_Z = 1 / 200

DIMS = [12, 24, 36]
POSES = [1]
POSES.extend([i*10 for i in range(1, 5)])
POSES.extend([i*50 for i in range(1, 21)])
