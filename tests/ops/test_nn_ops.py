import unittest
import numpy as np

from parameterized import parameterized

from tensorgrad.tensor import Tensor
from tensorgrad.const import DTYPE, BACKEND, OP
from ..util import require_torch, check_tensors, generate_cases

torch = require_torch()

OPS_TESTED = (
    OP.RELU,
    OP.SIGMOID,
)
SHAPES_TESTED = (
    (1,),
    (10,),
    (100,),
    (1000,),
    (5, 10),
    (50, 100),
    (5, 50, 100),
)
BACKENDS_TESTED = (
    BACKEND.NUMPY,
)
DTYPES_TESTED = (
    DTYPE.FLOAT32,
    # DTYPE.FLOAT64,
)

CASES = generate_cases(OPS_TESTED, SHAPES_TESTED, BACKENDS_TESTED, DTYPES_TESTED)


class TestBinaryOps(unittest.TestCase):

    @parameterized.expand(
        generate_cases(
            (
                OP.RELU,
                OP.SIGMOID
            ),
            (
                (1,),
                (10,),
                (100,),
                (1000,),
                (5, 10),
                (50, 100),
                (5, 50, 100),
            ),
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
    )
    def test_activations(self, op, shape, backend, dtype):
        name = f'{op}::{shape}::{backend}::{dtype}'
        method = self._op_to_method(op)

        _x = np.random.uniform(-1.0, 1.0, size=shape)
        # make sure having at least single element less than zero when testing relu-like ops.
        if op == OP.RELU:
            _x = _x.ravel()
            _x[0] = -1.0
            _x = _x.reshape(*shape)
        _x = _x.tolist()
    
        x = Tensor(_x, name='x', dtype=dtype, backend=backend, requires_grad=True)
        y = getattr(x, method)().sum()
        y.backward()

        tdtype = getattr(torch, dtype.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        ty = getattr(tx, method)().sum()
        ty.backward()

        self._check_tensors(ty, y, msg=f'{name}@forward')
        self._check_tensors(tx.grad, x.grad, msg=f'{name}@x_grad')

    @parameterized.expand(
        generate_cases(
            [0],
            [
                (1,),
                (10,),
                (100,),
                (1000,),
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [0, 1],
            [
                (2, 3),
                (200, 300),
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [0, 1, 2],
            [
                (2, 3, 4),
                (20, 30, 40),
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [0, 1, 2, 3],
            [
                (2, 3, 4, 5),
                (10, 20, 30, 40),
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
    )
    def test_softmax(self, dim, shape, backend, dtype):
        name = f'softmax::{dim}::{shape}::{backend}::{dtype}'

        _x = np.random.uniform(-1.0, 1.0, size=shape).tolist()
    
        x = Tensor(_x, name='x', dtype=dtype, backend=backend, requires_grad=True)
        y = x.softmax(dim).log().sum()
        y.backward()

        tdtype = getattr(torch, dtype.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        ty = tx.softmax(dim).log().sum()
        ty.backward()

        self._check_tensors(ty, y, msg=f'{name}@forward')
        self._check_tensors(tx.grad, x.grad, msg=f'{name}@x_grad')

    def _check_tensors(self, a, b, tol=1e-5, msg=''):
        self.assertTrue(check_tensors(a.tolist(), b.tolist(), tol=tol, show_diff=False), msg=msg)

    def _op_to_method(self, op):
        return {
            OP.RELU: 'relu',
            OP.SIGMOID: 'sigmoid',
        }[op]
