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
        self.helper._test_weight_and_bias_module(
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
        self.helper._test_weight_and_bias_module(
            test_name=name,
            module='Conv2d',
            input_shape=shape,
            torch_kwargs=kwargs,
            tensorgrad_kwargs=kwargs,
            tol=1e-4,
        )
    
    @parameterized.expand([
        [(2, 3, 32, 32), 2, None, 0],
        [(2, 3, 32, 32), 2, 2, 0],
        [(2, 3, 32, 32), (2, 2), (2, 2), 0],
        [(2, 3, 32, 32), 2, (2, 2), 1],
        [(2, 3, 32, 32), 2, (2, 2), (1, 1)],
        [(2, 3, 32, 32), (4, 4), 4, 0],
    ])
    def test_max_pool2d(
        self,
        shape,
        kernel_size,
        stride,
        padding
    ):
        kwargs = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
        }
        name = str(kwargs)
        self.helper._test_pooling_module(
            test_name=name,
            module='MaxPool2d',
            input_shape=shape,
            torch_kwargs=kwargs,
            tensorgrad_kwargs=kwargs,
            tol=1e-4,
        )
    
    @parameterized.expand([
        [(2, 3, 32, 32), 2, None, 0],
        [(2, 3, 32, 32), 2, 2, 0],
        [(2, 3, 32, 32), (2, 2), (2, 2), 0],
        [(2, 3, 32, 32), 2, (2, 2), 1],
        [(2, 3, 32, 32), 2, (2, 2), (1, 1)],
        [(2, 3, 32, 32), (4, 4), 4, 0],
    ])
    def test_avg_pool2d(
        self,
        shape,
        kernel_size,
        stride,
        padding
    ):
        kwargs = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
        }
        name = str(kwargs)
        self.helper._test_pooling_module(
            test_name=name,
            module='AvgPool2d',
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
        self.helper._test_weight_and_bias_module(
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
        self.helper._test_weight_and_bias_module(
            test_name=name,
            module='BatchNorm2d',
            input_shape=shape,
            torch_kwargs=kwargs,
            tensorgrad_kwargs=kwargs,
            tol=1e-4,
        )
    
    @parameterized.expand([
        [(2, 4, 8), 1],
        [(2, 4, 8), 2],
        [(2, 4, 8), 4],
        [(8, 32, 128), 8],
    ])
    def test_multihead_attention(self, shape, num_heads):
        embed_dim = shape[-1]
        _q = np.random.normal(size=shape)
        _k = np.random.normal(size=shape)
        _v = np.random.normal(size=shape)

        tdtype = getattr(torch, DTYPE.value)
        tq = torch.tensor(_q, dtype=tdtype, requires_grad=True)
        tk = torch.tensor(_k, dtype=tdtype, requires_grad=True)
        tv = torch.tensor(_v, dtype=tdtype, requires_grad=True)
        tm = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, bias=False)
        to, _ = tm(tq, tk, tv)
        self.helper._backward_torch(to)

        q = tensorgrad.Tensor(_q, dtype=DTYPE, device=DEVICE, requires_grad=True)
        k = tensorgrad.Tensor(_k, dtype=DTYPE, device=DEVICE, requires_grad=True)
        v = tensorgrad.Tensor(_v, dtype=DTYPE, device=DEVICE, requires_grad=True)
        m = tensorgrad.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        m.init_from_torch(tm)
        o = m(q, k, v)
        self.helper._backward_tensorgrad(o)

        name = f'{shape}::{num_heads}'
        tol = 1e-4
        self.helper._check_tensors([
            [to, o, tol, f'{name}@forward'],
            [tq.grad, q.grad, tol, f'{name}@q_grad'],
            [tk.grad, k.grad, tol, f'{name}@k_grad'],
            [tv.grad, v.grad, tol, f'{name}@v_grad'],
            [tm.in_proj_weight.grad.chunk(3)[0], m.q_weight.grad, tol, f'{name}@q_w_grad'],
            [tm.in_proj_weight.grad.chunk(3)[1], m.k_weight.grad, tol, f'{name}@k_w_grad'],
            [tm.in_proj_weight.grad.chunk(3)[2], m.v_weight.grad, tol, f'{name}@v_w_grad'],
            [tm.out_proj.weight.grad, m.o_weight.grad, tol, f'{name}@o_w_grad'],
        ])



class Helper(unittest.TestCase):

    def _test_weight_and_bias_module(
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
        m.init_from_torch(tm)
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
    
    def _test_pooling_module(
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
        o = m(x)
        self._backward_tensorgrad(o)

        self._check_tensors([
            [to, o, tol, f'{test_name}@forward'],
            [tx.grad, x.grad, tol, f'{test_name}@x_grad'],
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
