import unittest

import numpy as np
from parameterized import parameterized

from tests.const import DTYPE, DEVICE
from tests.util import require_torch
from tests.helper import CommonHelper

import tensorgrad
torch = require_torch()
torch.manual_seed(0)


class TestOptim(unittest.TestCase):

    def setUp(self) -> None:
        self.helper = Helper()

    @parameterized.expand([
        [(8, 16), 1, 0.1, 0.0],
        [(8, 16), 10, 0.1, 0.0],
        [(8, 16), 10, 0.1, 0.9],
        [(8, 16), 10, 0.01, 0.9],
    ])
    def test_sgd(self, shape, num_steps, lr, momentum):
        self.helper._test_optim(
            module='SGD',
            shape=shape,
            num_steps=num_steps,
            kwargs={'lr': lr, 'momentum': momentum}
        )
    
    @parameterized.expand([
        [(8, 16), 1],
        [(8, 16), 10],
    ])
    def test_adam(self, shape, num_steps):
        self.helper._test_optim(
            module='Adam',
            shape=shape,
            num_steps=num_steps,
        )


class Helper(unittest.TestCase):
    helper = CommonHelper()

    def _test_optim(self, module, shape, num_steps, kwargs=None):
        kwargs = kwargs or {}
        name = f'{module}::{shape}::{num_steps}::{kwargs}'

        _x = np.random.normal(size=shape)

        tdtype = getattr(torch, DTYPE.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        tx = torch.nn.Parameter(tx)
        toptim = getattr(torch.optim, module)
        toptim = toptim([tx], **kwargs)

        x = tensorgrad.nn.Parameter(tx.detach().numpy(), dtype=DTYPE, device=DEVICE, requires_grad=True)
        optim = getattr(tensorgrad.optim, module)
        optim = optim([x], **kwargs)

        tol = 1e-5
        for _ in range(num_steps):
            toptim.zero_grad()
            tf = tx.exp().mean()
            tf.backward()
            toptim.step()

            optim.zero_grad()
            f = x.exp().mean()
            f.backward()
            optim.step()
            
            self.helper._check_tensors([
                [tf, f, tol, f'forward@{name}'],
                [tx.grad, x.grad, tol, f'x_grad@{name}'],
            ])
