from hashing_utils import compactBit
from parameter import BATCH_SIZE
import torch
from tqdm import tqdm


def get_bit(num_train, num_test, TARGET_DIM, anchor_loader, test_loader, encoder):
  trainY = torch.empty(num_train, TARGET_DIM)

  testY = torch.empty(num_test, TARGET_DIM)

  encoder.cpu()


  print('making trainY')
  for step, (x, _) in tqdm(enumerate(anchor_loader)):
    encoded = encoder(x)
    trainY[(step*BATCH_SIZE):((step+1)*BATCH_SIZE)] = encoded

  print('making testY')
  for step, (x, y) in enumerate(test_loader):
    encoded = encoder(x)
    testY[(step*BATCH_SIZE):((step+1)*BATCH_SIZE)] = encoded


  trainY = trainY.detach().numpy()
  testY = testY.detach().numpy()

  trainY = compactBit(trainY)
  testY = compactBit(testY)

  return trainY, testY
