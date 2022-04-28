import torch
import numpy as np

from wea_nf.layers.actnorm import ActNorm


def test_initialization():
    a = ActNorm(dim=2)

    x = np.array([[1, 2], [2, 6], [7, 3]], dtype=np.float32)

    x_new, logdet = a.forward(torch.from_numpy(x))
    x_new = x_new.detach().cpu().numpy()
    logetd = logdet.detach().cpu().numpy()

    np.testing.assert_array_almost_equal(x_new.mean(axis=0), np.zeros((2,)))
    np.testing.assert_array_almost_equal(x_new.std(axis=0), np.ones((2,)))
    np.testing.assert_array_equal(logetd.shape, np.array([3]))
