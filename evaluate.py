from hashing_utils import recall_precision, recall_precision5, area_RP

def evaluate(ntrain, ntest, trainY, testY, trainLabel, testLable, Wtrue, POSES):
  precision, recall = recall_precision(trainY, testY, trainLabel, testLable, Wtrue)
  pre, rec = recall_precision5(trainY, testY, trainLabel, testLable, POSES, Wtrue)
  mAP = area_RP(recall, precision)

  return precision, recall, pre, rec, mAP

