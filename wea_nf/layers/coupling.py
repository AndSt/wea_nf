import torch
import torch.nn as nn

from wea_nf.layers.actnorm import ActNorm


class CouplingLayer(nn.Module):
    def __init__(self, s, t, dim: int, mask=None):
        super().__init__()
        self.t = t
        self.s = s
        self.dim = dim
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.norm = ActNorm(dim=dim)

    def forward(self, x):

        masked_x = x * self.mask
        s = (1 - self.mask) * self.s(masked_x)
        t = (1 - self.mask) * self.t(masked_x)

        y = masked_x + (1 - self.mask) * (x * torch.exp(s) + t)

        logdet = torch.sum(s, -1)
        y, ld = self.norm(y)
        logdet += ld
        return y, logdet

    def inverse(self, y):
        masked_y = y * self.mask

        s = self.s(masked_y)
        t = self.t(masked_y)

        x = masked_y + (1 - self.mask) * ((y - t) * torch.exp(-s))
        return x


class LinearCouplingLayer(CouplingLayer):
    def __init__(self, dim: int = 2, hidden_dim: int = 768, mask: torch.Tensor = None, dropout: float = 0.3):
        s = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )
        t = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )
        super().__init__(s=s, t=t, dim=dim, mask=mask)
