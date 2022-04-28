import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    """
    Code adapted from: https://github.com/chrischute/real-nvp/blob/master/util/norm_util.py
    """

    def __init__(self, dim, eps=1e-6, decay=0.1):
        super(BatchNorm, self).__init__()
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))
        self.decay = decay

    def forward(self, x: torch.Tensor):
        # Get mean and variance per channel
        if self.training:
            used_mean = x.mean(dim=0)
            used_var = x.var(dim=0, unbiased=False)

            # Update variables
            self.running_mean = self.running_mean - self.decay * (self.running_mean - used_mean)
            self.running_var = self.running_var - self.decay * (self.running_var - used_var)
        else:
            used_mean = self.running_mean
            used_var = self.running_var

        used_var += self.eps
        x = (x - used_mean) / torch.sqrt(used_var)
        logdet = - 0.5 * torch.log(used_var).sum()
        logdet = logdet.repeat(x.shape[0])

        return x, logdet
