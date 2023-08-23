import unittest

import numpy as np
from parameterized import parameterized

from tests.util import require_torch, check_tensors, generate_cases, get_device, get_dtype

import tensorgrad
torch = require_torch()

DEVICE = get_device()
DTYPE = get_dtype()


class TestOps(unittest.TestCase):

    def setUp(self) -> None:
        self.helper = Helper()

    @parameterized.expand([
        [(128,)],
        [(32, 128)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_add(self, shape):
        self.helper._test_binary_op(shape=shape, method='__add__')
    
    @parameterized.expand([
        [(128,)],
        [(32, 128)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_sub(self, shape):
        self.helper._test_binary_op(shape=shape, method='__sub__')
    
    @parameterized.expand([
        [(128,)],
        [(32, 128)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_mul(self, shape):
        self.helper._test_binary_op(shape=shape, method='__mul__')
    
    @parameterized.expand([
        [(128,)],
        [(32, 128)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_div(self, shape):
        self.helper._test_binary_op(shape=shape, method='__truediv__')
    
    @parameterized.expand([
        [(128,), 2.7],
        [(128,), 0.5],
        [(128,), 0],
        [(32, 128,), 2.7],
        [(32, 128,), 0.5],
        [(32, 128,), 0],
    ])
    def test_pow(self, shape, p):
        x = np.random.uniform(0.0, 1.0, size=shape)
        self.helper._test_unary_op(shape=shape, method='__pow__', args=(p,), x=x)
    
    @parameterized.expand([
        [(128,)],
        [(32, 128)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_log(self, shape):
        x = np.random.uniform(0.0, 1.0, size=shape)
        self.helper._test_unary_op(shape=shape, method='log', args=(), x=x)
    
    @parameterized.expand([
        [(128,)],
        [(32, 128)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_exp(self, shape):
        self.helper._test_unary_op(shape=shape, method='exp', args=())

    @parameterized.expand([
        [(1, 128), 0],
        [(128, 1), 1],
        [(1, 128), -2],
        [(128, 1), -1],
        [(1, 16, 32), 0],
        [(8, 1, 32),  1],
        [(8, 16, 1),  2],
        [(1, 128), 0],
        [(128, 1), 1],
        [(1, 8, 16, 32), 0],
        [(4, 1, 16, 32), 1],
        [(4, 8, 1, 32),  2],
        [(4, 8, 16, 1),  3],
    ])
    def test_squeeze(self, shape, dim):
        self.helper._test_unary_op(shape=shape, method='squeeze', args=(dim,))
    
    @parameterized.expand([
        [(128,), 0],
        [(128,), 1],
        [(128,), -1],
        [(32, 64), 0],
        [(32, 64), 1],
        [(32, 64), 2],
        [(32, 64), -1],
        [(8, 16, 32), 0],
        [(8, 16, 32), 1],
        [(8, 16, 32), 2],
        [(8, 16, 32), 3],
        [(8, 16, 32), -1],
        [(4, 8, 16, 32), 0],
        [(4, 8, 16, 32), 1],
        [(4, 8, 16, 32), 2],
        [(4, 8, 16, 32), 3],
        [(4, 8, 16, 32), 4],
        [(4, 8, 16, 32), -1],
    ])
    def test_unsqueeze(self, shape, dim):
        self.helper._test_unary_op(shape=shape, method='unsqueeze', args=(dim,))

    @parameterized.expand([
        [(6,), (2, 3)],
        [(3, 1), (1, 3)],
        [(3, 4), (2, 6)],
        [(2, 3, 4), (2, 12)],
        [(2, 12), (2, 3, 4)],
        [(2, 3, 4, 5), (6, 4, 5)],
        [(2, 3, 4, 5), (-1, 5)],
        [(2, 3, 4, 5), (24, -1)],
        [(2, 3, 4, 5), (-1,)],
    ])
    def test_reshape(self, shape_a, shape_b):
        self.helper._test_unary_op(shape=shape_a, method='reshape', args=(shape_b,))

    @parameterized.expand([
        [(4, 1), (0, 1)],
        [(1, 4), (1, 0)],
        [(2, 3), (0, 1)],
        [(2, 3), (1, 0)],
        [(2, 3, 4), (0, 1, 2)],
        [(2, 3, 4), (1, 0, 2)],
        [(2, 3, 4), (1, 2, 0)],
        [(2, 3, 4), (0, 2, 1)],
        [(2, 3, 4), (2, 0, 1)],
        [(2, 3, 4), (2, 1, 0)],
        [(2, 3, 4, 5), (0, 1, 2, 3)],
        [(2, 3, 4, 5), (1, 0, 2, 3)],
        [(2, 3, 4, 5), (0, 2, 1, 3)],
        [(2, 3, 4, 5), (1, 2, 0, 3)],
        [(2, 3, 4, 5), (2, 0, 1, 3)],
        [(2, 3, 4, 5), (2, 1, 0, 3)],
        [(2, 3, 4, 5), (0, 1, 3, 2)],
        [(2, 3, 4, 5), (1, 0, 3, 2)],
        [(2, 3, 4, 5), (0, 2, 3, 1)],
        [(2, 3, 4, 5), (1, 2, 3, 0)],
        [(2, 3, 4, 5), (2, 0, 3, 1)],
        [(2, 3, 4, 5), (2, 1, 3, 0)],
        [(2, 3, 4, 5), (0, 3, 1, 2)],
        [(2, 3, 4, 5), (1, 3, 0, 2)],
        [(2, 3, 4, 5), (0, 3, 2, 1)],
        [(2, 3, 4, 5), (1, 3, 2, 0)],
        [(2, 3, 4, 5), (2, 3, 0, 1)],
        [(2, 3, 4, 5), (2, 3, 1, 0)],
        [(2, 3, 4, 5), (3, 0, 1, 2)],
        [(2, 3, 4, 5), (3, 1, 0, 2)],
        [(2, 3, 4, 5), (3, 0, 2, 1)],
        [(2, 3, 4, 5), (3, 1, 2, 0)],
        [(2, 3, 4, 5), (3, 2, 0, 1)],
        [(2, 3, 4, 5), (3, 2, 1, 0)],
    ])
    def test_permute(self, shape, dims):
        self.helper._test_unary_op(shape=shape, method='permute', args=(dims,))

    @parameterized.expand([
        [(3, 4), 0],
        [(3, 4), (slice(None), 0)],
        [(3, 4), (0, slice(None))],
        [(4, 5), (slice(1, 3), slice(None))],
        [(4, 5), (slice(None), slice(2, 4))],
        [(4, 5), (1, slice(2, 4))],
        [(4, 5), (slice(1, 3), 4)],
        [(4, 5), (1, 4)],
        [(2, 3, 4), 0],
        [(2, 3, 4), (slice(None), slice(1, 2))],
        [(2, 3, 4), (0, slice(1, 2), -1)],
        [(2, 3, 4), (slice(0, 1), slice(1, 2), slice(2, 3))],
    ])
    def test_select(self, shape, slice_):
        self.helper._test_unary_op(shape=shape, method='__getitem__', args=(slice_,))


class Helper(unittest.TestCase):

    def _test_unary_op(self, shape, method, x=None, args=None, kwargs=None):
        args = args or ()
        kwargs = kwargs or {}
        _x = x if x is not None else np.random.normal(0.0, 1.0, size=shape)
        
        x = tensorgrad.Tensor(_x, device=DEVICE, dtype=DTYPE, requires_grad=True, name='x')
        o = getattr(x, method)(*args, **kwargs)
        self._backward_tensorgrad(o)

        tdtype = getattr(torch, x.dtype.value)
        tx = torch.tensor(_x, requires_grad=True, dtype=tdtype)
        to = getattr(tx, method)(*args, **kwargs)
        self._backward_torch(to)

        name = f'{shape}::{method}::{args}'
        self.assertTrue(check_tensors(to.tolist(), o.tolist(), show_diff=False), msg=f'{name}@forward')
        self.assertTrue(check_tensors(tx.grad.tolist(), x.grad.tolist(), show_diff=False), msg=f'{name}@x_grad')

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
