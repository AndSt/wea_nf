import torch
import torch.nn as nn


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.inv_log_std = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.is_initialized = self.register_buffer("is_initialized", torch.zeros(1))

    def initialize(self, x: torch.Tensor):
        self.bias.data = torch.mean(x, dim=0)
        self.inv_log_std.data = - torch.log(torch.std(x, dim=0, unbiased=False) + 1e-12)
        self.is_initialized = torch.ones(1)

    def forward(self, x) -> [torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.initialize(x)

        x = x - self.bias
        x = x * torch.exp(self.inv_log_std)

        # derivative in each dimension equal
        logdet = torch.sum(self.inv_log_std, dim=0)
        logdet = logdet.repeat(x.shape[0])

        return x, logdet
