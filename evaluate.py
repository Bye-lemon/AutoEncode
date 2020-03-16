from hashing_utils import recall_precision, recall_precision5, area_RP, hamming_dist

def evaluate(ntrain, ntest, trainY, testY, trainLabel, testLable, Wtrue, POSES):
  print('calculating hamming_dist')
  hamdis = hamming_dist(trainY, testY)
  print('calculating precision')
  precision, recall = recall_precision(trainY, testY, trainLabel, testLable, Wtrue, hamdis)
  print('calculating pre')
  pre, rec = recall_precision5(trainY, testY, trainLabel, testLable, POSES, Wtrue, hamdis)
  mAP = area_RP(recall, precision)

  return precision, recall, pre, rec, mAP

