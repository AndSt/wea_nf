import math

import torch.nn as nn
import torch.nn.functional as F


class Tanh(nn.Module):
    """Taken from https://github.com/kamenbliznashki/normalizing_flows
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # derivation of logdet:
        # d/dx tanh = 1 / cosh^2; cosh = (1 + exp(-2x)) / (2*exp(-x))
        # log d/dx tanh = - 2 * log cosh = -2 * (x - log 2 + log(1 + exp(-2x)))
        logdet = -2 * (x - math.log(2) + F.softplus(-2*x))

        return x.tanh(), logdet