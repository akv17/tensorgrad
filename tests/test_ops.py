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
        self.helper._test_binary_op(a_shape=shape, b_shape=shape, method='__add__')
    
    @parameterized.expand([
        [(128,)],
        [(32, 128)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_sub(self, shape):
        self.helper._test_binary_op(a_shape=shape, b_shape=shape, method='__sub__')
    
    @parameterized.expand([
        [(128,)],
        [(32, 128)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_mul(self, shape):
        self.helper._test_binary_op(a_shape=shape, b_shape=shape, method='__mul__')
    
    @parameterized.expand([
        [(128,)],
        [(32, 128)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_div(self, shape):
        self.helper._test_binary_op(a_shape=shape, b_shape=shape, method='__truediv__')
    
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

    @parameterized.expand([
        [(128,), None],
        [(32, 64), None],
        [(32, 64), 0],
        [(32, 64), 1],
        [(32, 64), -1],
        [(32, 64), -2],
        [(8, 16, 32), None],
        [(8, 16, 32), 0],
        [(8, 16, 32), 1],
        [(8, 16, 32), 2],
        [(4, 8, 16, 32), None],
        [(4, 8, 16, 32), 0],
        [(4, 8, 16, 32), 1],
        [(4, 8, 16, 32), 2],
        [(4, 8, 16, 32), 3],
    ])
    def test_sum_reduce(self, shape, dim):
        self.helper._test_unary_op(shape=shape, method='sum', args=(dim,))
    
    @parameterized.expand([
        [(128,), None],
        [(32, 64), None],
        [(32, 64), 0],
        [(32, 64), 1],
        [(32, 64), -1],
        [(32, 64), -2],
        [(8, 16, 32), None],
        [(8, 16, 32), 0],
        [(8, 16, 32), 1],
        [(8, 16, 32), 2],
        [(4, 8, 16, 32), None],
        [(4, 8, 16, 32), 0],
        [(4, 8, 16, 32), 1],
        [(4, 8, 16, 32), 2],
        [(4, 8, 16, 32), 3],
    ])
    def test_mean_reduce(self, shape, dim):
        self.helper._test_unary_op(shape=shape, method='mean', args=(dim,))
    
    @parameterized.expand([
        [(128,)],
        [(32, 64)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_relu(self, shape):
        self.helper._test_unary_op(shape=shape, method='relu')
    
    @parameterized.expand([
        [(128,)],
        [(32, 64)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_sigmoid(self, shape):
        self.helper._test_unary_op(shape=shape, method='sigmoid')

    @parameterized.expand([
        [(2, 4), (4, 2)],
        [(2, 2), (2, 2)],
        [(4, 1), (1, 4)],
        [(1, 4), (4, 1)],
        [(128, 256), (256, 128)],
        [(2, 3, 4), (2, 4, 3)],
        [(20, 30, 40), (20, 40, 30)],
        [(2, 3, 4, 5), (2, 3, 5, 4)],
        [(20, 30, 40, 50), (20, 30, 50, 40)],
        [(2, 3, 4), (4, 5)],
        [(2, 5, 3, 4), (4, 5)],
        [(3, 2, 5, 3, 4), (4, 5)],
        [(4, 5), (2, 5, 4)],
        [(4, 5), (2, 3, 5, 4)],
        [(4, 5), (3, 2, 4, 5, 4)],
    ])
    def test_matmul(self, a_shape, b_shape):
        self.helper._test_binary_op(a_shape=a_shape, b_shape=b_shape, method='matmul', tol=1e-4)

    @parameterized.expand([
        [(128,), 0],
        [(128,), -1],
        [(32, 64), 0],
        [(32, 64), 1],
        [(32, 64), -1],
        [(32, 64), -2],
        [(8, 16, 32), 0],
        [(8, 16, 32), 1],
        [(8, 16, 32), 2],
        [(4, 8, 16, 32), 0],
        [(4, 8, 16, 32), 1],
        [(4, 8, 16, 32), 2],
        [(4, 8, 16, 32), 3],
    ])
    def test_softmax(self, shape, dim):
        self.helper._test_unary_op(shape=shape, method='softmax', args=(dim,))

    @parameterized.expand([
        [2, (5, 5), (3, 3), 4, 8, False, (1, 1), (0, 0)],
        [2, (5, 5), (3, 3), 4, 8, True, (1, 1), (0, 0)],
        [2, (5, 5), (1, 1), 4, 8, True, (1, 1), (0, 0)],
        [2, (9, 9), (5, 5), 4, 8, True, (1, 1), (0, 0)],
        
        [2, (5, 5), (3, 3), 4, 8, True, (2, 2), (1, 1)],
        [2, (5, 5), (3, 3), 4, 8, True, (1, 2), (1, 1)],
        [2, (5, 5), (3, 3), 4, 8, True, (2, 1), (1, 1)],
        
        [2, (5, 5), (3, 3), 4, 8, True, (2, 1), (1, 2)],
        [2, (5, 5), (3, 3), 4, 8, True, (1, 2), (1, 2)],
        [2, (5, 5), (3, 3), 4, 8, True, (2, 1), (2, 1)],
        [2, (5, 5), (3, 3), 4, 8, True, (1, 2), (2, 1)],
        
        [4, (64, 64), (3, 3), 3, 16, True, (1, 1), (2, 2)],
        [4, (32, 32), (3, 3), 64, 32, True, (1, 1), (2, 2)],
    ])
    def test_conv2d(
        self,
        batch_size,
        input_size,
        kernel_size,
        in_channels,
        out_channels,
        bias,
        stride,
        padding
    ):
        _x = np.random.normal(size=(batch_size, in_channels, *input_size))
        _k = np.random.normal(size=(out_channels, in_channels, *kernel_size))
        _b = np.random.normal(size=(out_channels,)) if bias else None

        x = tensorgrad.Tensor(_x, device=DEVICE, dtype=DTYPE, requires_grad=True, name='x')
        k = tensorgrad.Tensor(_k, device=DEVICE, dtype=DTYPE, requires_grad=True, name='k')
        b = tensorgrad.Tensor(_b, device=DEVICE, dtype=DTYPE, requires_grad=True, name='b') if bias else None
        o = x.conv2d(kernel=k, bias=b, stride=stride, padding=padding)
        self.helper._backward_tensorgrad(o)

        tdtype = getattr(torch, x.dtype.value)
        tx = torch.tensor(_x, requires_grad=True, dtype=tdtype)
        tk = torch.tensor(_k, requires_grad=True, dtype=tdtype)
        tb = torch.tensor(_b, requires_grad=True, dtype=tdtype) if bias else None
        to = torch.nn.functional.conv2d(tx, tk, tb, stride=stride, padding=padding)
        self.helper._backward_torch(to)

        name = f'{batch_size}::{input_size}::{kernel_size}::{in_channels}::{out_channels}::{bias}::{stride}::{padding}'
        tol = 1e-3
        self.assertTrue(check_tensors(to.tolist(), o.tolist(), tol=tol, show_diff=False), msg=f'{name}@forward')
        self.assertTrue(check_tensors(tx.grad.tolist(), x.grad.tolist(), tol=tol, show_diff=True), msg=f'{name}@x_grad')
        self.assertTrue(check_tensors(tk.grad.tolist(), k.grad.tolist(), tol=tol, show_diff=True), msg=f'{name}@k_grad')
        if bias:
            self.assertTrue(check_tensors(tb.grad.tolist(), b.grad.tolist(), tol=tol, show_diff=False), msg=f'{name}@b_grad')


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

        name = f'{shape}::{method}::{args}::{kwargs}'
        self.assertTrue(check_tensors(to.tolist(), o.tolist(), show_diff=False), msg=f'{name}@forward')
        self.assertTrue(check_tensors(tx.grad.tolist(), x.grad.tolist(), show_diff=False), msg=f'{name}@x_grad')

    def _test_binary_op(self, a_shape, b_shape, method, tol=1e-5):
        _a = np.random.normal(size=a_shape)
        _b = np.random.normal(size=b_shape)
        
        a = tensorgrad.Tensor(_a, device=DEVICE, dtype=DTYPE, requires_grad=True, name='a')
        b = tensorgrad.Tensor(_b, device=DEVICE, dtype=DTYPE, requires_grad=True, name='b')
        o = getattr(a, method)(b)
        self._backward_tensorgrad(o)

        tdtype = getattr(torch, a.dtype.value)
        ta = torch.tensor(_a, requires_grad=True, dtype=tdtype)
        tb = torch.tensor(_b, requires_grad=True, dtype=tdtype)
        to = getattr(ta, method)(tb)
        self._backward_torch(to)

        name = f'{a_shape}::{b_shape}::{method}'
        self.assertTrue(check_tensors(to.tolist(), o.tolist(), tol=tol, show_diff=False), msg=f'{name}@forward')
        self.assertTrue(check_tensors(ta.grad.tolist(), a.grad.tolist(), tol=tol, show_diff=False), msg=f'{name}@a_grad')
        self.assertTrue(check_tensors(tb.grad.tolist(), b.grad.tolist(), tol=tol, show_diff=False), msg=f'{name}@b_grad')

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
