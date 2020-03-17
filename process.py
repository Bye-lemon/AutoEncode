def process(encoder, enc_optimizer, decoder, dec_optimizer, mseLoss_fun, tripletLoss_fun, train_loader, EPOCH, BATCH_SIZE, LAMBDA_T):
  for epoch in range(EPOCH):
    for step, x in enumerate(train_loader):
      b_x = x.view(BATCH_SIZE, 3, 1, 28, 28).cuda()
      b_y = x.view(BATCH_SIZE, 3, 1, 28, 28).cuda()

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

      loss = encode_loss + LAMBDA_T * triplet_loss
      
      enc_optimizer.zero_grad()
      dec_optimizer.zero_grad()
      loss.backward()
      enc_optimizer.step()
      dec_optimizer.step()

      if step % 100 == 0:
        print('Epoch: {}, train_loss: {}'.format(epoch, loss.cpu().data.numpy()))
        print(anc_encoded.size())
