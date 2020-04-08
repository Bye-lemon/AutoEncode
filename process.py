import torch
from parameter import *

def process(encoder, enc_optimizer, decoder, dec_optimizer, mseLoss_fun, tripletLoss_fun, train_loader):
  for epoch in range(EPOCH):
    for step, x in enumerate(train_loader):
      b_x = x.view(BATCH_SIZE, 3, 1, 28, 28)
      b_y = x.view(BATCH_SIZE, 3, 1, 28, 28)
      if torch.cuda.is_available():
        b_x = b_x.cuda()
        b_y = b_y.cuda()

      # batchsize * num_bits
      anc_encoded = encoder(b_x[:, 0])
      anc_decoded = decoder(anc_encoded)
      pos_encoded = encoder(b_x[:, 1])
      pos_decoded = decoder(pos_encoded)
      neg_encoded = encoder(b_x[:, 2])
      neg_decoded = decoder(neg_encoded)
      # (anc_encoded, anc_decoded, pos_encoded,
      # pos_decoded, neg_encoded, neg_decoded) = autocoder(b_x)

      encode_loss = mseLoss_fun(anc_decoded, b_y[:, 0]) + \
                    mseLoss_fun(pos_decoded, b_y[:, 1]) + \
                    mseLoss_fun(neg_decoded, b_y[:, 2])

      triplet_loss = tripletLoss_fun(anc_encoded, pos_encoded, neg_encoded)

      uncorrelated_loss = torch.sum(torch.abs(
        anc_encoded.mm(anc_encoded.transpose(0, 1)) + \
        pos_encoded.mm(pos_encoded.transpose(0, 1)) + \
        neg_encoded.mm(neg_encoded.transpose(0, 1)) - \
        3 * torch.eye(BATCH_SIZE)
      ))

      var_loss = torch.sum(torch.var(anc_encoded, 0)) + \
                 torch.sum(torch.var(pos_encoded, 0)) + \
                 torch.sum(torch.var(neg_encoded, 0))

      zero_loss = torch.sum(
        (torch.abs(anc_encoded) - torch.ones_like(anc_encoded)) ** 2 + \
        (torch.abs(pos_encoded) - torch.ones_like(pos_encoded)) ** 2 + \
        (torch.abs(neg_encoded) - torch.ones_like(neg_encoded)) ** 2
      )

      loss = encode_loss + \
             LAMBDA_T * triplet_loss + \
             LAMBDA_U * uncorrelated_loss + \
             LAMBDA_V * var_loss + \
             LAMBDA_Z * zero_loss
      
      enc_optimizer.zero_grad()
      dec_optimizer.zero_grad()
      loss.backward()
      enc_optimizer.step()
      dec_optimizer.step()

      if step % 100 == 0:
        print('Epoch: {}, train_loss: {}'.format(epoch, loss.cpu().data.numpy()))
        print('encode_loss: {}'.format(encode_loss.cpu().data.numpy()))
        print('triplet_loss: {}'.format(triplet_loss.cpu().data.numpy()))
        print('uncorrelated_loss: {}'.format(uncorrelated_loss.cpu().data.numpy()))
        print('var_loss: {}'.format(var_loss.cpu().data.numpy()))
        print('zero_loss: {}'.format(zero_loss.cpu().data.numpy()))
