import unittest
import numpy as np

from parameterized import parameterized

from tensorgrad.tensor import Tensor
from tensorgrad.const import DTYPE, BACKEND, OP
from ..util import require_torch, check_tensors, generate_cases

torch = require_torch()

OPS_TESTED = (
    OP.SUM_REDUCE,
    OP.MEAN_REDUCE,
)
SHAPES_TESTED = (
    (1,),
    (10,),
    (100,),
    (1000,),
)
BACKENDS_TESTED = (
    BACKEND.NUMPY,
)
DTYPES_TESTED = (
    DTYPE.FLOAT32,
    DTYPE.FLOAT64,
)

CASES = generate_cases(OPS_TESTED, SHAPES_TESTED, BACKENDS_TESTED, DTYPES_TESTED)


class TestUnaryOps(unittest.TestCase):

    @parameterized.expand(CASES)
    def test(self, op, shape, backend, dtype):
        name = f'{op}::{shape}::{backend}::{dtype}'
        method = self._op_to_method(op)

        _x = np.random.random(shape).tolist()

        x = Tensor(_x, name='x', dtype=dtype, backend=backend, requires_grad=True)
        y = getattr(x, method)().sum()
        y.backward()

        tdtype = getattr(torch, dtype.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        ty = getattr(tx, method)().sum()
        ty.backward()

        self._check_tensors(ty, y, msg=f'{name}@forward')
        self._check_tensors(tx.grad, x.grad, msg=f'{name}@x_grad')

    @parameterized.expand(POW_CASES)
    def test_pow(self, value, shape, backend, dtype):
        name = f'pow::{shape}::{backend}::{dtype}'

        _x = np.random.random(shape).tolist()

        x = Tensor(_x, name='x', dtype=dtype, backend=backend, requires_grad=True)
        y = (x ** value).sum()
        y.backward()

        tdtype = getattr(torch, dtype.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        ty = (tx ** value).sum()
        ty.backward()

        self._check_tensors(ty, y, msg=f'{name}@forward')
        self._check_tensors(tx.grad, x.grad, msg=f'{name}@x_grad')

    def _check_tensors(self, a, b, tol=1e-5, msg=''):
        self.assertTrue(check_tensors(a.tolist(), b.tolist(), tol=tol), msg=msg)

    def _op_to_method(self, op):
        return {
            OP.SUM_REDUCE: 'sum',
            OP.MEAN_REDUCE: 'mean',
        }[op]
