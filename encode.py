import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

from hashing_utils import *
from model import Encoder, Decoder
from parameter import *
from load_data import load_data
from process import process
from get_bits import get_bit
from evaluate import evaluate
from plot_graph import plot_graph

anchor_data, train_data, test_data, num_train, num_test = load_data(DATASETPATH, DOWNLOAD)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
anchor_loader = Data.DataLoader(dataset=anchor_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

trainLabel = anchor_data.targets.numpy()
testLabel = test_data.targets.numpy()

# for evaluation
Wtrue = generate_Wtrue(num_train, num_test, trainLabel, testLabel)
precision_dims = []
recall_dims = []
pre_dims = []
rec_dims = []
mAP_dims =[]

for TARGET_DIM in DIMS:
  # define net work
  # autocoder = AutoEncoder(TARGET_DIM)
  encoder = Encoder(TARGET_DIM).cuda()
  decoder = Decoder(TARGET_DIM).cuda()

  enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
  dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)
  mseLoss_fun = nn.MSELoss()
  tripletLoss_fun = nn.TripletMarginLoss(margin=MARGIN, p=1)

  # training network
  process(encoder, enc_optimizer, decoder, dec_optimizer, mseLoss_fun, tripletLoss_fun, train_loader, EPOCH, BATCH_SIZE, LAMBDA_T)

  torch.save(encoder, './nets/encoder.pkl')
  torch.save(decoder, './nets/decoder.pkl')

  input()
  # get bit
  trainY, testY = get_bit(num_train, num_test, TARGET_DIM, anchor_loader, test_loader, encoder)

  # for evaluation

  precision, recall, pre, rec, mAP = evaluate(num_train, num_test, trainY, testY, trainLabel, testLabel, Wtrue, POSES)

  precision_dims.append(precision)
  recall_dims.append(recall)
  pre_dims.append(pre)
  rec_dims.append(rec)
  mAP_dims.append(mAP)

plot_graph(DATASETNAME, DIMS, LAMBDA_T, POSES, precision_dims, recall_dims, pre_dims, rec_dims, mAP_dims)
