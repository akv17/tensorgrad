import unittest

import numpy as np
from parameterized import parameterized

from tests.util import require_torch, check_tensors, generate_cases, get_device, get_dtype

import tensorgrad
torch = require_torch()

DEVICE = get_device()
DTYPE = get_dtype()


class TestOps(unittest.TestCase):

    @parameterized.expand([
        [(128,)],
        [(32, 128)],
        [(8, 16, 32)],

    ])
    def test_add(self, shape):
        self._test_binary_op(shape=shape, method='__add__')

    def _test_binary_op(self, shape, method):
        _a = np.random.normal(size=shape)
        _b = np.random.normal(size=shape)
        
        a = tensorgrad.Tensor(_a, device=DEVICE, dtype=DTYPE, requires_grad=True, name='a')
        b = tensorgrad.Tensor(_b, device=DEVICE, dtype=DTYPE, requires_grad=True, name='b')
        o = getattr(a, method)(b)
        self._backward_tensorgrad(o)

        tdtype = getattr(torch, a.dtype.value)
        ta = torch.tensor(_a, requires_grad=True, dtype=tdtype)
        tb = torch.tensor(_b, requires_grad=True, dtype=tdtype)
        to = getattr(ta, method)(tb)
        self._backward_torch(to)

        name = f'{shape}::{method}'
        self.assertTrue(check_tensors(to.tolist(), o.tolist(), show_diff=False), msg=f'{name}@forward')
        self.assertTrue(check_tensors(ta.grad.tolist(), a.grad.tolist(), show_diff=False), msg=f'{name}@a_grad')
        self.assertTrue(check_tensors(tb.grad.tolist(), b.grad.tolist(), show_diff=False), msg=f'{name}@b_grad')

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
