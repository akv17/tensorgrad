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
        kwargs = {'in_features': shape[-1], 'out_features': out_features, 'bias': bias}
        name = str(kwargs)
        self.helper._test_module_with_weight_and_bias(
            test_name=name,
            module='Linear',
            input_shape=shape,
            torch_kwargs=kwargs,
            tensorgrad_kwargs=kwargs,
            tol=1e-5,
        )

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
        shape = (batch_size, in_channels, *input_size)
        kwargs = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias,
        }
        name = str(kwargs)
        self.helper._test_module_with_weight_and_bias(
            test_name=name,
            module='Conv2d',
            input_shape=shape,
            torch_kwargs=kwargs,
            tensorgrad_kwargs=kwargs,
            tol=1e-4,
        )
    
    @parameterized.expand([
        [(2, 8)],
        [(16, 32)],
        [(128, 256)],
    ])
    def test_batch_norm1d(self, shape):
        kwargs = {'num_features': shape[-1]}
        name = str(kwargs)
        self.helper._test_module_with_weight_and_bias(
            test_name=name,
            module='BatchNorm1d',
            input_shape=shape,
            torch_kwargs=kwargs,
            tensorgrad_kwargs=kwargs,
            tol=1e-4,
        )

    @parameterized.expand([
        [(2, 3, 4, 4)],
        [(4, 8, 16, 16)],
        [(8, 16, 32, 32)],
    ])
    def test_batch_norm2d(self, shape):
        kwargs = {'num_features': shape[1]}
        name = str(kwargs)
        self.helper._test_module_with_weight_and_bias(
            test_name=name,
            module='BatchNorm2d',
            input_shape=shape,
            torch_kwargs=kwargs,
            tensorgrad_kwargs=kwargs,
            tol=1e-4,
        )


class Helper(unittest.TestCase):

    def _test_module_with_weight_and_bias(
        self,
        test_name,
        module,
        input_shape,
        torch_kwargs,
        tensorgrad_kwargs,
        tol=1e-5,
    ):
        _x = np.random.normal(size=input_shape)
        
        tdtype = getattr(torch, DTYPE.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        tm = getattr(torch.nn, module)(**torch_kwargs)
        to = tm(tx)
        self._backward_torch(to)

        x = tensorgrad.Tensor(_x, dtype=DTYPE, device=DEVICE, name='x', requires_grad=True)
        m = getattr(tensorgrad.nn, module)(**tensorgrad_kwargs)
        m.weight = tensorgrad.Tensor(tm.weight.detach().numpy(), dtype=DTYPE, device=DEVICE, name='w', requires_grad=True)
        if tm.bias is not None:
            m.bias = tensorgrad.Tensor(tm.bias.detach().numpy(), dtype=DTYPE, device=DEVICE, name='b', requires_grad=True)
        o = m(x)
        self._backward_tensorgrad(o)

        self._check_tensors([
            [to, o, tol, f'{test_name}@forward'],
            [tx.grad, x.grad, tol, f'{test_name}@x_grad'],
            [tm.weight.grad, m.weight.grad, tol, f'{test_name}@w_grad'],
        ])
        if tm.bias is not None:
            self._check_tensors([
                [tm.bias.grad, m.bias.grad, tol, f'{test_name}@b_grad'],
            ])

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
