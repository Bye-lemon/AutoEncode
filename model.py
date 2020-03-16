import torch.nn as nn

class AutoEncoder(nn.Module):
  def __init__(self, target_dim):
    super(AutoEncoder, self).__init__()

    self.target_dim = target_dim

    self.encoder = nn.Sequential(
      nn.Linear(28 * 28, 128),
      nn.Tanh(),
      nn.Linear(128, 64),
      nn.Tanh(),
      nn.Linear(64, self.target_dim),
      nn.Sigmoid(),
    )
    self.decoder = nn.Sequential(
      nn.Linear(self.target_dim, 64),
      nn.Tanh(),
      nn.Linear(64, 128),
      nn.Tanh(),
      nn.Linear(128, 28 * 28),
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