import unittest

import numpy as np
from parameterized import parameterized

import tensorgrad
from tensorgrad.const import BACKEND, DTYPE
from ..util import require_torch, check_tensors, generate_cases

torch = require_torch()

BACKENDS_TESTED = [BACKEND.NUMPY]
DTYPES_TESTED = [DTYPE.FLOAT32]


class TestModules(unittest.TestCase):

    @parameterized.expand(
        generate_cases(
            [
                (1, 3),
                (3, 1),
                (2, 3),
                (16, 32),
                (2, 3, 4),
                (2, 3, 4, 5),
            ],
            [1, 4, 8, 64, 128],
            [True, False],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
    )
    def test_linear(self, input_size, linear_size, bias, backend, dtype):
        name = f'{input_size}::{linear_size}::{bias}::{backend}::{dtype}'

        _x = np.random.uniform(-1.0, 1.0, size=input_size).tolist()
        in_features = input_size[-1]
        out_features = linear_size

        tdtype = getattr(torch, dtype.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        tlinear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        ty = tlinear(tx).sum()
        ty.backward()

        linear = tensorgrad.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        linear.initialize(weight=tlinear.weight.data.numpy(), bias=tlinear.bias.data.numpy() if bias else None)
        x = tensorgrad.Tensor(_x, name='x', dtype=dtype, backend=backend, requires_grad=True)
        y = linear(x).sum()
        y.backward()

        self._check_tensors(ty, y, msg=f'{name}@forward')
        self._check_tensors(tlinear.weight.grad, linear.weight.grad, msg=f'{name}@w_grad')
        if bias:
            self._check_tensors(tlinear.bias.grad, linear.bias.grad, msg=f'{name}@b_grad')
    
    def _check_tensors(self, a, b, tol=1e-5, msg=''):
        self.assertTrue(check_tensors(a.tolist(), b.tolist(), tol=tol), msg=msg)
