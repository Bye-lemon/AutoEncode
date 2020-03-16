DATASETNAME = 'FashionMnist'
DATASETPATH = './fashion_mnist/'
DOWNLOAD = False
if not(os.path.exists(DATASETPATH)) or not os.listdir(DATASETPATH):
  # nor dir or dir is empty
  DOWNLOAD = True

EPOCH = 10
BATCH_SIZE = 50
LR = 0.005

TARGET_DIM = 8
MARGIN = 1
LAMBDA_T = 3

DIMS = [12, 24, 36, 48]
POSES = [1]
POSES.extend([i*10 for i in range(1, 5)])
POSES.extend([i*50 for i in range(1, 21)])