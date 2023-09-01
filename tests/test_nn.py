import os
import unittest

import numpy as np
from parameterized import parameterized

from tests.util import require_torch, check_tensors, generate_cases, get_device, get_dtype

import tensorgrad
torch = require_torch()

DEVICE = get_device()
DTYPE = get_dtype()
SHOW_DIFF = os.getenv('TESTS_SHOW_DIFF') == '1'
RENDER = os.getenv('TESTS_RENDER') == '1'


class TestNN(unittest.TestCase):

    def setUp(self) -> None:
        self.helper = Helper()

    @parameterized.expand([
        [(2, 8)],
        [(16, 32)],
        [(128, 256)],
    ])
    def test_batch_norm1d(self, shape):
        _x = np.random.normal(size=shape)
        num_features = _x.shape[1]
        
        tdtype = getattr(torch, DTYPE.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        tm = torch.nn.BatchNorm1d(num_features=num_features)
        to = tm(tx)
        self.helper._backward_torch(to)

        x = tensorgrad.Tensor(_x, dtype=DTYPE, device=DEVICE, name='x', requires_grad=True)
        m = tensorgrad.nn.BatchNorm1D(num_features=num_features)
        m.weight = tensorgrad.Tensor(tm.weight.detach().numpy(), dtype=DTYPE, device=DEVICE, name='w', requires_grad=True)
        m.bias = tensorgrad.Tensor(tm.bias.detach().numpy(), dtype=DTYPE, device=DEVICE, name='b', requires_grad=True)
        o = m(x)
        self.helper._backward_tensorgrad(o)

        tol = 1e-5
        name = f'{shape}'
        self.helper._check_tensors([
            [to, o, tol, f'{name}@forward'],
            [tx.grad, x.grad, tol, f'{name}@x_grad'],
        ])


class Helper(unittest.TestCase):

    def _check_tensors(self, pairs):
        for tt, t, tol, name in pairs:
            self.assertTrue(check_tensors(tt.tolist(), t.tolist(), tol=tol, show_diff=SHOW_DIFF), msg=name)

    def _backward_tensorgrad(self, tensor):
        r = tensor.arange(tensor.numel()).reshape(tensor.shape) + 1.0
        norm = r.data.max().tolist()
        r = r / norm
        o = (tensor * r).sum()
        o.backward()
    
    def _backward_torch(self, tensor):
        r = torch.arange(tensor.numel()).reshape(tensor.shape) + 1.0
        norm = r.data.max().tolist()
        r = r / norm
        o = (tensor * r).sum()
        o.backward()
