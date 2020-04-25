import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

from hashing_utils import *
from parameter import *
from model import Encoder, Decoder, ResEncoder, ResDecoder, DenseEncoder, DenseDecoder
from load_data import load_data_no_triplet
from process import process_no_triplet
from get_bits import get_bit
from evaluate import evaluate
from plot_graph import plot_graph


train_data, test_data, num_train, num_test = load_data_no_triplet(DATASETNAME, DATASETPATH, DOWNLOAD)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

if DATASETNAME == 'FashionMnist':
  trainLabel = train_data.targets.numpy()
  testLabel = test_data.targets.numpy()
elif DATASETNAME == 'CIFAR10':
  trainLabel = train_data.targets
  testLabel = test_data.targets


# for evaluation
# Wtrue = generate_Wtrue(num_train, num_test, trainLabel, testLabel)
precision_dims = []
recall_dims = []
pre_dims = []
rec_dims = []
mAP_dims =[]

for TARGET_DIM in DIMS:
  # define net work
  # autocoder = AutoEncoder(TARGET_DIM)
  encoder = Encoder(TARGET_DIM, CHANNEL)
  decoder = Decoder(TARGET_DIM, CHANNEL)
  if torch.cuda.is_available():
    encoder = encoder.cuda()
    decoder = decoder.cuda()

  enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
  dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)
  mseLoss_fun = nn.MSELoss()

  # training network
  process_no_triplet(encoder, enc_optimizer, decoder, dec_optimizer, mseLoss_fun, train_loader, TARGET_DIM)

  # torch.save(encoder.state_dict(), './nets/encoder_params.pkl')
  # torch.save(decoder.state_dict(), './nets/decoder_params.pkl')

  # input()
  # get bit
  trainY, testY = get_bit(num_train, num_test, TARGET_DIM, train_loader, test_loader, encoder)

  # for evaluation

  precision, recall, pre, rec, mAP = evaluate(num_train, num_test, trainY, testY, trainLabel, testLabel, Wtrue, POSES)

  precision_dims.append(precision)
  recall_dims.append(recall)
  pre_dims.append(pre)
  rec_dims.append(rec)
  mAP_dims.append(mAP)

plot_graph(DATASETNAME, DIMS, LAMBDA_T, POSES, precision_dims, recall_dims, pre_dims, rec_dims, mAP_dims)
