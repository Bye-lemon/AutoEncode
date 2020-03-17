from hashing_utils import compactBit
import torch


def get_bit(num_train, num_test, TARGET_DIM, anchor_data, test_data, encoder):
  trainY = torch.empty(num_train, TARGET_DIM)

  testY = torch.empty(num_test, TARGET_DIM)

  for index, (x, label) in enumerate(anchor_data):
    x = x.view(1, 1, 28, 28).cuda()
    encoded = encoder(x)
    trainY[index] = encoded[0]

  print('after mking trainY')

  for index, (x, label) in enumerate(test_data):
    x = x.view(1, 1, 28, 28).cuda()
    encoded = encoder(x)
    testY[index] = encoded[0]

  print('after mking testY')

  trainY = trainY.detach().cpu().numpy()
  testY = testY.detach().cpu().numpy()

  trainY = compactBit(trainY)
  testY = compactBit(testY)

  return trainY, testY
