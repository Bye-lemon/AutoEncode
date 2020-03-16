import torch.nn as nn

class AutoEncoder(nn.Module):
  def __init__(self, target_dim):
    super(AutoEncoder, self).__init__()

    self.target_dim = target_dim

    self.encoder = nn.Sequential(
      nn.Linear(28 * 28, 392),
      nn.Tanh(),
      nn.Linear(392, 196),
      nn.Tanh(),
      nn.Linear(196, 128),
      nn.Tanh(),
      nn.Linear(128, 64),
      nn.Tanh(),
      nn.Linear(64, TARGET_DIM),
      nn.Sigmoid(),
    )
    self.decoder = nn.Sequential(
      nn.Linear(TARGET_DIM, 64),
      nn.Tanh(),
      nn.Linear(64, 128),
      nn.Tanh(),
      nn.Linear(128, 196),
      nn.Tanh(),
      nn.Linear(196, 392),
      nn.Tanh(),
      nn.Linear(392, 28 * 28),
      nn.Sigmoid(),
    )

  def forward(self, x):
    anc_encode = self.encoder(x[:, 0, :])
    anc_decode = self.decoder(anc_encode)
    pos_encode = self.encoder(x[:, 1, :])
    pos_decode = self.decoder(pos_encode)
    neg_encode = self.encoder(x[:, 2, :])
    neg_decode = self.decoder(neg_encode)
    return anc_encode, anc_decode, pos_encode, pos_decode, neg_encode, neg_decode