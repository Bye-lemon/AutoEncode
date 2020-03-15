import numpy as np
import matplotlib.pyplot as plt

def compactBit(A):
  B = np.zeros((A.shape[0], A.shape[1]))
  B[A>0.5] = 1
  return B

def hamming_dist(X, Y):
    hashbits = X.shape[1]
    Xint = (2 * X.astype('int8')) - 1
    Yint = (2 * Y.astype('int8')) - 1
    hamdis = hashbits - ((hashbits + Xint.dot(Yint.T)) / 2)
    return hamdis

def recall_precision(trainY, testY, traingnd, testgnd):
    # make sure trangnd and testgnd are flattened
    testgnd = testgnd.ravel()
    traingnd = traingnd.ravel()
    ntrain = trainY.shape[0]
    ntest = testY.shape[0]

    Wture = np.zeros((ntrain, ntest))
    for i in range(ntrain):
        for j in range(ntest):
            if traingnd[i] == testgnd[j]:
                Wture[i, j] = 1

    total_good_pairs = Wture.sum()

    hamdis = hamming_dist(trainY, testY)
    max_hamm = int(hamdis.max())
    
    precision = np.zeros(max_hamm)
    recall = np.zeros(max_hamm)

    for redius in range(max_hamm):
        TestTrue = np.zeros((ntrain, ntest))
        TestTrue[hamdis <= redius+0.00001] = 1

        retrieved_good_pairs = Wture[TestTrue>0].sum()
        retrieved_pairs = TestTrue.sum()

        precision[redius] = retrieved_good_pairs / (retrieved_pairs + 1e-3)
        recall[redius] = retrieved_good_pairs / total_good_pairs

    return precision, recall

def recall_precision5(trainY, testY, traingnd, testgnd, pos):
    # make sure trangnd and testgnd are flattened
    testgnd = testgnd.ravel()
    traingnd = traingnd.ravel()
    ntrain = trainY.shape[0]
    ntest = testY.shape[0]
    npos = len(pos)

    Wture = np.zeros((ntrain, ntest))
    for i in range(ntrain):
        for j in range(ntest):
            if traingnd[i] == testgnd[j]:
                Wture[i, j] = 1

    hamdis = hamming_dist(trainY, testY)

    for i in range(ntest):
        index_ = hamdis[:, i].argsort()
        Wture[:, i] = Wture[index_, i]

    total_good_pairs = Wture.sum()

    recall = np.zeros(npos)
    precision = np.zeros(npos)

    # pylint: disable=unpacking-non-sequence
    for i in range(npos):
        g = pos[i]
        retrieved_good_pairs = Wture[:g, :].sum()
        row, col = Wture[:g, :].shape
        total_pairs = row * col
        recall[i] = retrieved_good_pairs / total_good_pairs
        precision[i] = retrieved_good_pairs / (total_pairs + 1e-3)

    return precision, recall

def area_RP(recall, precision):
    xx, index_ = np.unique(recall, return_index=True)
    yy = precision[index_]
    for iii in range(len(xx)):
        ic = len(xx) - iii - 1
        if yy[ic] < 0:
            yy[ic] = yy[ic + 1]
    area = 0
    for i in range(len(xx)-1):
        subarea = 0.5 * (xx[i+1] - xx[i])*(yy[i+1] + yy[i])
        area += subarea
    return area

def plot_precision_recall(precision, recall):
    plt.subplot(141)
    plt.plot(recall, precision, 'bo')
    plt.xlabel('recall')
    plt.ylabel('precision')

def plot_recall_number(recall, poses):
    plt.subplot(142)
    plt.plot(poses, recall,  'ro')
    plt.xlabel('the number of retrieved samples')
    plt.ylabel('recall')

def plot_precision_number(precision, poses):
    plt.subplot(143)
    plt.plot(poses, precision, 'g*--')
    plt.xlabel('the number of retrieved samples')
    plt.ylabel('precision')

def plot_mAP(mAP, dims):
    plt.subplot(144)
    plt.plot(dims, mAP, 'cd--')
    plt.xlabel('the number of bits')
    plt.ylabel('mAP')