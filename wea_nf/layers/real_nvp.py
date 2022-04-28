import numpy as np

import torch
from torch import nn
from torch import distributions

from wea_nf.layers.coupling import LinearCouplingLayer
from wea_nf.layers.actnorm import ActNorm


def generate_masks(dim: int, depth: int = 2):
    masks = []
    for i in list(range(depth)):
        if i == 0:
            mask = np.zeros((dim,))
            mask[0:int(dim / 2)] = 1
            masks.append(mask)
        else:
            mask = np.ones((dim,)) - masks[i - 1]
            masks.append(mask)
    return masks


class RealNVP(nn.Module):
    """
    Implementation adapted from: https://github.com/xqding/RealNVP
    """

    def __init__(self, dim: int, hidden_dim: int, depth: int = 2, **kwargs):
        super().__init__(**kwargs)

        self.prior = distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))

        self.dim = dim
        self.hidden_dim = hidden_dim

        masks = generate_masks(dim=dim, depth=depth)
        self.masks = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m), requires_grad=False) for m in masks]
        )

        self.affine_couplings = nn.ModuleList(
            [LinearCouplingLayer(dim=dim, hidden_dim=hidden_dim, mask=self.masks[i]) for i in range(len(self.masks))]
        )
        self.norm = ActNorm(dim)

    def forward(self, x):
        y, logdet_tot = self.norm(x)

        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot += logdet

        return y, logdet_tot

    def log_prob(self, x):
        z, logdet = self.forward(x)
        device = z.device
        z = z.to(torch.device("cpu"))
        log_prob = self.prior.log_prob(z).to(device)
        return log_prob, logdet
