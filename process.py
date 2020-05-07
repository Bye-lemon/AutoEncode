import torch
from parameter import *

def process(encoder, enc_optimizer, decoder, dec_optimizer, mseLoss_fun, tripletLoss_fun, train_loader, n_dim):
  for epoch in range(EPOCH):
    for step, x in enumerate(train_loader):
      b_x = x
      b_y = x
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

      Eyes = 3 * BATCH_SIZE * torch.eye(n_dim)
      if torch.cuda.is_available():
        Eyes = Eyes.cuda()

      uncorrelated_loss = torch.sum(torch.abs(
        torch.abs(anc_encoded.transpose(0, 1).mm(anc_encoded)) + \
        torch.abs(pos_encoded.transpose(0, 1).mm(pos_encoded)) + \
        torch.abs(neg_encoded.transpose(0, 1).mm(neg_encoded)) - \
        Eyes
      ))

      var_loss = torch.sum(torch.var(anc_encoded, 0)) + \
                 torch.sum(torch.var(pos_encoded, 0)) + \
                 torch.sum(torch.var(neg_encoded, 0))

      zero_loss = ((
        torch.sum(anc_encoded) + \
        torch.sum(pos_encoded) + \
        torch.sum(neg_encoded)
      ) / (3 * BATCH_SIZE * n_dim) - 0.5) ** 2

      loss = encode_loss + \
             LAMBDA_T * triplet_loss + \
             LAMBDA_U * uncorrelated_loss - \
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

def process_no_triplet(encoder, enc_optimizer, decoder, dec_optimizer, mseLoss_fun, train_loader, n_dim):
  for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
      b_x = x
      b_y = x
      if torch.cuda.is_available():
        b_x = b_x.cuda()
        b_y = b_y.cuda()

      # batchsize * num_bits
      encoded = encoder(b_x)
      decoded = decoder(encoded)
      # (anc_encoded, anc_decoded, pos_encoded,
      # pos_decoded, neg_encoded, neg_decoded) = autocoder(b_x)

      encode_loss = mseLoss_fun(decoded, b_y)

      Eyes = BATCH_SIZE * torch.eye(n_dim)
      if torch.cuda.is_available():
        Eyes = Eyes.cuda()

      uncorrelated_loss = torch.sum(torch.abs(
        torch.abs(encoded.transpose(0, 1).mm(encoded)) - \
        Eyes
      ))

      var_loss = torch.sum(torch.var(encoded, 0))

      zero_loss = (
        torch.sum(encoded)
       / (BATCH_SIZE * n_dim)) ** 2

      loss = encode_loss + \
             LAMBDA_U * uncorrelated_loss - \
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
        print('uncorrelated_loss: {}'.format(uncorrelated_loss.cpu().data.numpy()))
        print('var_loss: {}'.format(var_loss.cpu().data.numpy()))
        print('zero_loss: {}'.format(zero_loss.cpu().data.numpy()))
