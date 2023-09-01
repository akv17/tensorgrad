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
        [(2, 8), 16, True],
        [(2, 8), 16, False],
        [(2, 8, 16), 32, True],
        [(2, 8, 16), 32, False],
        [(128, 256), 512, True],
    ])
    def test_linear(self, shape, out_features, bias):
        _x = np.random.normal(size=shape)
        in_features = _x.shape[-1]
        
        tdtype = getattr(torch, DTYPE.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        tm = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        to = tm(tx)
        self.helper._backward_torch(to)

        x = tensorgrad.Tensor(_x, dtype=DTYPE, device=DEVICE, name='x', requires_grad=True)
        m = tensorgrad.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        m.weight = tensorgrad.Tensor(tm.weight.detach().numpy(), dtype=DTYPE, device=DEVICE, name='w', requires_grad=True)
        if bias:
            m.bias = tensorgrad.Tensor(tm.bias.detach().numpy(), dtype=DTYPE, device=DEVICE, name='b', requires_grad=True)
        o = m(x)
        self.helper._backward_tensorgrad(o)

        tol = 1e-5
        name = f'{shape}'
        self.helper._check_tensors([
            [to, o, tol, f'{name}@forward'],
            [tx.grad, x.grad, tol, f'{name}@x_grad'],
            [tm.weight.grad, m.weight.grad, tol, f'{name}@w_grad'],
        ])
        if bias:
            self.helper._check_tensors([
                [tm.bias.grad, m.bias.grad, tol, f'{name}@b_grad'],
            ])

    @parameterized.expand([
        [2, (32, 32), (3, 3), 3, 16, False, 1, 0],
        [2, (32, 32), (3, 3), 3, 16, True, 1, 0],
        [2, (32, 32), 3, 3, 16, True, 1, 0],
        [2, (32, 32), 3, 3, 16, True, (2, 2), (1, 1)],
        [2, (32, 32), 3, 3, 16, True, 4, 2],
        [2, (32, 32), 3, 3, 16, True, 1, 'valid'],
        [2, (32, 32), 3, 3, 16, True, 1, 'same'],
        [2, (32, 32), 5, 3, 16, True, 1, 'same'],
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
        in_channels = in_channels
        out_channels = out_channels

        tdtype = getattr(torch, DTYPE.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        tm = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        to = tm(tx)
        self.helper._backward_torch(to)

        x = tensorgrad.Tensor(_x, device=DEVICE, dtype=DTYPE, requires_grad=True, name='x')
        m = tensorgrad.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        m.weight = tensorgrad.Tensor(tm.weight.detach().numpy(), dtype=DTYPE, device=DEVICE, name='w', requires_grad=True)
        if bias:
            m.bias = tensorgrad.Tensor(tm.bias.detach().numpy(), dtype=DTYPE, device=DEVICE, name='b', requires_grad=True)
        o = m(x)
        self.helper._backward_tensorgrad(o)

        name = f'{batch_size}::{input_size}::{kernel_size}::{in_channels}::{out_channels}::{bias}::{stride}::{padding}'
        tol = 1e-4
        self.helper._check_tensors([
            [to, o, tol, f'{name}@forward'],
            [tx.grad, x.grad, tol, f'{name}@x_grad'],
            [tm.weight.grad, m.weight.grad, tol, f'{name}@w_grad'],
        ])
        if bias:
            self.helper._check_tensors([
                [tm.bias.grad, m.bias.grad, tol, f'{name}@b_grad'],
            ])

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
        m = tensorgrad.nn.BatchNorm1d(num_features=num_features)
        m.weight = tensorgrad.Tensor(tm.weight.detach().numpy(), dtype=DTYPE, device=DEVICE, name='w', requires_grad=True)
        m.bias = tensorgrad.Tensor(tm.bias.detach().numpy(), dtype=DTYPE, device=DEVICE, name='b', requires_grad=True)
        o = m(x)
        self.helper._backward_tensorgrad(o)

        tol = 1e-5
        name = f'{shape}'
        self.helper._check_tensors([
            [to, o, tol, f'{name}@forward'],
            [tx.grad, x.grad, tol, f'{name}@x_grad'],
            [tm.weight.grad, m.weight.grad, tol, f'{name}@w_grad'],
            [tm.bias.grad, m.bias.grad, tol, f'{name}@b_grad'],
        ])

    @parameterized.expand([
        [(2, 3, 4, 4)],
        [(4, 8, 16, 16)],
        [(8, 16, 64, 64)],
    ])
    def test_batch_norm2d(self, shape):
        _x = np.random.normal(size=shape)
        num_features = _x.shape[1]

        tdtype = getattr(torch, DTYPE.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        tm = torch.nn.BatchNorm2d(num_features=num_features)
        to = tm(tx)
        self.helper._backward_torch(to)

        x = tensorgrad.Tensor(_x, dtype=DTYPE, device=DEVICE, name='x', requires_grad=True)
        m = tensorgrad.nn.BatchNorm2d(num_features=num_features)
        m.weight = tensorgrad.Tensor(tm.weight.detach().numpy(), dtype=DTYPE, device=DEVICE, name='w', requires_grad=True)
        m.bias = tensorgrad.Tensor(tm.bias.detach().numpy(), dtype=DTYPE, device=DEVICE, name='b', requires_grad=True)
        o = m(x)
        self.helper._backward_tensorgrad(o)

        tol = 1e-4
        name = f'{shape}'
        self.helper._check_tensors([
            [to, o, tol, f'{name}@forward'],
            [tx.grad, x.grad, tol, f'{name}@x_grad'],
            [tm.weight.grad, m.weight.grad, tol, f'{name}@w_grad'],
            [tm.bias.grad, m.bias.grad, tol, f'{name}@b_grad'],
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
