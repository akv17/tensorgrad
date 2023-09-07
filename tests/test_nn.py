import os
import unittest

import numpy as np
from parameterized import parameterized

from tests.util import require_torch, check_tensors, generate_cases, get_device, get_dtype

import tensorgrad
torch = require_torch()
torch.manual_seed(0)

DEVICE = get_device()
DTYPE = get_dtype()
SHOW_DIFF = os.getenv('TESTS_SHOW_DIFF') == '1'
RENDER = os.getenv('TESTS_RENDER') == '1'


class TestNN(unittest.TestCase):

    def setUp(self) -> None:
        self.helper = Helper()
    
    @parameterized.expand([
        [(128,)],
        [(32, 64)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_relu(self, shape):
        name = f'{shape}'
        self.helper._test_module_without_params(
            test_name=name,
            module='ReLU',
            input_shape=shape,
            torch_kwargs={},
            tensorgrad_kwargs={},
        )
    
    @parameterized.expand([
        [(128,)],
        [(32, 64)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_sigmoid(self, shape):
        name = f'{shape}'
        self.helper._test_module_without_params(
            test_name=name,
            module='Sigmoid',
            input_shape=shape,
            torch_kwargs={},
            tensorgrad_kwargs={},
        )
    
    @parameterized.expand([
        [(128,)],
        [(32, 64)],
        [(8, 16, 32)],
        [(4, 8, 16, 32)],
    ])
    def test_identity(self, shape):
        name = f'{shape}'
        self.helper._test_module_without_params(
            test_name=name,
            module='Identity',
            input_shape=shape,
            torch_kwargs={},
            tensorgrad_kwargs={},
        )

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
        [(2, 3, 32, 32), 2, None, 0],
        [(2, 3, 32, 32), 2, 2, 0],
        [(2, 3, 32, 32), (2, 2), (2, 2), 0],
        [(2, 3, 32, 32), 2, (2, 2), 1],
        [(2, 3, 32, 32), 2, (2, 2), (1, 1)],
        [(2, 3, 32, 32), (4, 4), 4, 0],
        [(4, 16, 26, 26), (26, 26), None, 0],
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
        self.helper._test_module_without_params(
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
        self.helper._test_module_without_params(
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
    
    @parameterized.expand([
        [(2, 4), 1],
        [(2, 4, 8), 1],
        [(2, 4, 8), 2],
        [(2, 3, 16, 16), 3],
    ])
    def test_layer_norm(self, shape, dims):
        kwargs = {'normalized_shape': shape[-dims:]}
        name = str(kwargs)
        self.helper._test_module_with_weight_and_bias(
            test_name=name,
            module='LayerNorm',
            input_shape=shape,
            torch_kwargs=kwargs,
            tensorgrad_kwargs=kwargs,
            tol=1e-4,
        )
    
    @parameterized.expand([
        [(2, 4, 8), (2, 4, 8), 1, None],
        [(2, 4, 8), (2, 4, 8), 2, None],
        [(2, 4, 8), (2, 4, 8), 2, 2],
        [(2, 4, 8), (2, 4, 8), 2, 3],

        [(2, 4, 8), (2, 6, 8), 1, None],
        [(2, 4, 8), (2, 6, 8), 2, None],
        [(2, 4, 8), (2, 6, 8), 2, 2],
        [(2, 4, 8), (2, 6, 8), 2, 3],

        [(8, 32, 128), (8, 32, 128), 8, 3],
    ])
    def test_multihead_attention(self, q_shape, kv_shape, num_heads, attn_mask_dim):
        self.helper._test_multihead_attention(
            q_shape=q_shape,
            kv_shape=kv_shape,
            num_heads=num_heads,
            attn_mask_dim=attn_mask_dim
        )

    @parameterized.expand([
        [(2, 4), 8, 16],
        [(2, 4, 8), 8, 16],
        [(2, 4, 8, 16), 8, 16],
    ])
    def test_embedding(self, shape, num_embeddings, embedding_dim):
        self.helper._test_embedding(
            shape=shape,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        )

    @parameterized.expand([
        [(2, 4), 0.1],
        [(2, 4, 8), 0.1],
        [(2, 3, 16, 16), 0.1],
    ])
    def test_dropout(self, shape, p):
        self.helper._test_dropout(shape=shape, p=p)
    
    @parameterized.expand([
        [(2, 4)],
        [(2, 4, 8)],
        [(2, 4, 8, 16)],
    ])
    def test_flatten(self, shape):
        name = f'{shape}'
        self.helper._test_module_without_params(
            test_name=name,
            module='Flatten',
            input_shape=shape,
            torch_kwargs={},
            tensorgrad_kwargs={},
        )

    @parameterized.expand([
        [(2, 1)],
        [(2, 4)],
        [(64, 32)],
    ])
    def test_cross_entropy_loss(self, shape):
        num_classes = shape[-1]
        logits = np.random.normal(size=(shape))
        targets = np.random.randint(0, num_classes, size=(shape[0],))
        name = f'{shape}::{num_classes}'
        self.helper._test_loss(
            test_name=name,
            module='CrossEntropyLoss',
            outputs=logits,
            targets=targets,
            targets_as_int=True
        )
    
    @parameterized.expand([
        [(2, 1)],
        [(2, 4)],
        [(64, 32)],
    ])
    def test_mse_loss(self, shape):
        logits = np.random.normal(size=shape)
        targets = np.random.normal(size=shape)
        name = f'{shape}'
        self.helper._test_loss(
            test_name=name,
            module='MSELoss',
            outputs=logits,
            targets=targets,
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
    
    def _test_module_without_params(
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

    def _test_multihead_attention(self, q_shape, kv_shape, num_heads, attn_mask_dim):
        batch_size, q_seq_len, embed_dim = q_shape
        batch_size, kv_seq_len, embed_dim = kv_shape
        _q = np.random.normal(size=q_shape)
        _k = np.random.normal(size=kv_shape)
        _v = np.random.normal(size=kv_shape)
        if attn_mask_dim is not None:
            _attn_mask = np.ones((q_seq_len, kv_seq_len))
            _attn_mask = np.triu(_attn_mask, 1).astype('bool')
            if attn_mask_dim == 3:
                _attn_mask = np.expand_dims(_attn_mask, 0)
                _attn_mask = np.tile(_attn_mask, [batch_size, 1, 1])
        else:
            _attn_mask = None

        tdtype = getattr(torch, DTYPE.value)
        tq = torch.tensor(_q, dtype=tdtype, requires_grad=True)
        tk = torch.tensor(_k, dtype=tdtype, requires_grad=True)
        tv = torch.tensor(_v, dtype=tdtype, requires_grad=True)
        # reshape 3d mask to the shape expected by torch which is (bs*num_heads, q_seq_len, kv_seq_len).
        # this way torch allows to specify unique mask per head.
        # tensorgrad doesn't allow that and broadcasts the same mask over all heads.
        _tattn_mask = _attn_mask
        if _attn_mask is not None:
            if _attn_mask.ndim == 3:
                _tattn_mask = np.tile(_attn_mask, [num_heads, 1, 1])
            else:
                _tattn_mask = _attn_mask
        tattn_mask = torch.tensor(_tattn_mask, dtype=torch.bool, requires_grad=False) if _tattn_mask is not None else None
        tm = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, bias=False)
        to, _ = tm(tq, tk, tv, attn_mask=tattn_mask)
        self._backward_torch(to)

        q = tensorgrad.Tensor(_q, dtype=DTYPE, device=DEVICE, requires_grad=True)
        k = tensorgrad.Tensor(_k, dtype=DTYPE, device=DEVICE, requires_grad=True)
        v = tensorgrad.Tensor(_v, dtype=DTYPE, device=DEVICE, requires_grad=True)
        attn_mask = tensorgrad.Tensor(_attn_mask, dtype=DTYPE.BOOL, device=DEVICE, requires_grad=False) if _attn_mask is not None else None
        m = tensorgrad.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        m.init_from_torch(tm)
        o = m(q, k, v, attn_mask=attn_mask)
        self._backward_tensorgrad(o)

        name = f'{q_shape}::{num_heads}'
        tol = 1e-4
        self._check_tensors([
            [to, o, tol, f'{name}@forward'],
            [tq.grad, q.grad, tol, f'{name}@q_grad'],
            [tk.grad, k.grad, tol, f'{name}@k_grad'],
            [tv.grad, v.grad, tol, f'{name}@v_grad'],
            [tm.in_proj_weight.grad.chunk(3)[0], m.q_weight.grad, tol, f'{name}@q_w_grad'],
            [tm.in_proj_weight.grad.chunk(3)[1], m.k_weight.grad, tol, f'{name}@k_w_grad'],
            [tm.in_proj_weight.grad.chunk(3)[2], m.v_weight.grad, tol, f'{name}@v_w_grad'],
            [tm.out_proj.weight.grad, m.o_weight.grad, tol, f'{name}@o_w_grad'],
        ])

    def _test_embedding(self, shape, num_embeddings, embedding_dim):
        _x = np.random.randint(0, num_embeddings, size=shape)

        tx = torch.tensor(_x, dtype=torch.int32, requires_grad=False)
        tm = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        to = tm(tx)
        self._backward_torch(to)

        x = tensorgrad.Tensor(_x, dtype=DTYPE.INT32, device=DEVICE, name='x', requires_grad=False)
        m = tensorgrad.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        m.init_from_torch(tm)
        o = m(x)
        self._backward_tensorgrad(o)

        tol = 1e-5
        test_name = f'{shape}::{num_embeddings}::{embedding_dim}'
        self._check_tensors([
            [to, o, tol, f'{test_name}@forward'],
            [tm.weight.grad, m.weight.grad, tol, f'{test_name}@w_grad'],
        ])

    def _test_dropout(self, shape, p):
        _x = np.random.normal(size=shape)
        
        tdtype = getattr(torch, DTYPE.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        tm = torch.nn.Dropout(p)
        to = tm(tx)
        tmask = to == 0.0
        self._backward_torch(to)


        x = tensorgrad.Tensor(_x, dtype=DTYPE, device=DEVICE, name='x', requires_grad=True)
        m = tensorgrad.nn.Dropout(p)
        # monkey patch mask generation to apply exactly the same mask as applied by torch.
        mask = tmask.detach().cpu().numpy()
        mask = tensorgrad.Tensor(mask, dtype=DTYPE.BOOL, requires_grad=False)
        m._generate_mask = lambda __x: (~mask).float()
        o = m(x)
        self._backward_tensorgrad(o)

        tol = 1e-5
        test_name = f'{shape}::{p}'
        self._check_tensors([
            [to, o, tol, f'{test_name}@forward'],
            [tx.grad, x.grad, tol, f'{test_name}@x_grad'],
        ])

    def _test_loss(self, test_name, module, outputs, targets, targets_as_int=False):
        _outputs = outputs
        _targets = targets
        
        tdtype = getattr(torch, DTYPE.value)
        to = torch.tensor(_outputs, dtype=tdtype, requires_grad=True)
        ttdtype = torch.long if targets_as_int else tdtype
        tt = torch.tensor(_targets, dtype=ttdtype, requires_grad=False)
        tm = getattr(torch.nn, module)()
        tloss = tm(to, tt)
        tloss.backward()

        o = tensorgrad.Tensor(_outputs, dtype=DTYPE, device=DEVICE, requires_grad=True)
        tdtype = DTYPE.INT32 if targets_as_int else DTYPE
        t = tensorgrad.Tensor(_targets, dtype=tdtype, device=DEVICE, requires_grad=False)
        m = getattr(tensorgrad.nn, module)()
        loss = m(o, t)
        loss.backward()

        tol = 1e-5
        self._check_tensors([
            [tloss, loss, tol, f'{test_name}@forward'],
            [to.grad, o.grad, tol, f'{test_name}@o_grad'],
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
