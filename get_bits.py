from hashing_utils import compactBit
from parameter import BATCH_SIZE
import torch


def get_bit(num_train, num_test, TARGET_DIM, anchor_loader, test_loader, encoder):
  trainY = torch.empty(num_train, TARGET_DIM)

  testY = torch.empty(num_test, TARGET_DIM)

  print('making trainY')
  for step, x in enumerate(anchor_loader):
    x = x.view(1, 1, 28, 28).cuda()
    encoded = encoder(x)
    trainY[(step*BATCH_SIZE):((step+1)*BATCH_SIZE)] = encoded

  print('making testY')
  for step, x in enumerate(test_loader):
    x = x.view(1, 1, 28, 28).cuda()
    encoded = encoder(x)
    testY[(step*BATCH_SIZE):((step+1)*BATCH_SIZE)] = encoded


  trainY = trainY.detach().cpu().numpy()
  testY = testY.detach().cpu().numpy()

  trainY = compactBit(trainY)
  testY = compactBit(testY)

  return trainY, testY
