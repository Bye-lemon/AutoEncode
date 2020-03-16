from hashing_utils import compactBit

def get_bit(num_train, num_test, TARGET_DIM, anchor_data, test_data, autocoder):
  trainY = torch.empty(num_train, TARGET_DIM)
  trainLabel = anchor_data.targets.numpy()

  testY = torch.empty(num_test, TARGET_DIM)
  testLable = test_data.targets.numpy()

  for index, (x, label) in enumerate(anchor_data):
    x = x.view(-1, 28 * 28)
    encoded = autocoder.encoder(x)
    trainY[index] = encoded

  print('after mking trainY')

  for index, (x, label) in enumerate(test_data):
    x = x.view(-1, 28 * 28)
    encoded = autocoder.encoder(x)
    testY[index] = encoded

  print('after mking testY')

  trainY = trainY.detach().numpy()
  testY = testY.detach().numpy()

  trainY = compactBit(trainY)
  testY = compactBit(testY)

  return trainY, testY, trainLabel, testLable