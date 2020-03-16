def process(autocoder, optimizer, mseLoss_fun, tripletLoss_fun, train_loader, EPOCH, BATCH_SIZE, LAMBDA_T):
  for epoch in range(EPOCH):
    for step, x in enumerate(train_loader):
      b_x = x.view(BATCH_SIZE, 3, 28 * 28)
      b_y = x.view(BATCH_SIZE, 3, 28 * 28)

      (anc_encoded, anc_decoded, pos_encoded,
      pos_decoded, neg_encoded, neg_decoded) = autocoder(b_x)

      encode_loss = mseLoss_fun(anc_decoded, b_y[:, 0, :]) + \
                    mseLoss_fun(pos_decoded, b_y[:, 1, :]) + \
                    mseLoss_fun(neg_decoded, b_y[:, 2, :])
      triplet_loss = tripletLoss_fun(anc_encoded, pos_encoded, neg_encoded)

      loss = encode_loss + LAMBDA_T * triplet_loss
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if step % 100 == 0:
        print('Epoch: {}, train_loss: {}'.format(epoch, loss.data.numpy()))
