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
BACKENDS_TESTED = (
    BACKEND.NUMPY,
)
DTYPES_TESTED = (
    DTYPE.FLOAT32,
    DTYPE.FLOAT64,
)
CASES_1D = generate_cases(
    [None, 0],
    [
        (1,),
        (10,),
        (100,),
        (1000,),
    ],
    OPS_TESTED,
    BACKENDS_TESTED,
    DTYPES_TESTED
)
CASES_2D = generate_cases(
    [None, 0, 1],
    [
        (5, 10),
        (50, 100),
    ],
    OPS_TESTED,
    BACKENDS_TESTED,
    DTYPES_TESTED
)
CASES_3D = generate_cases(
    [None, 0, 1, 2],
    [
        (2, 5, 10),
        (5, 50, 100),
    ],
    OPS_TESTED,
    BACKENDS_TESTED,
    DTYPES_TESTED
)
CASES = CASES_1D + CASES_2D + CASES_3D


class TestReduceOps(unittest.TestCase):

    @parameterized.expand(CASES)
    def test(self, dim, shape, op, backend, dtype):
        name = f'{op}::{shape}::{dim}::{backend}::{dtype}'
        method = self._op_to_method(op)

        _x = np.random.random(shape).tolist()

        x = Tensor(_x, name='x', dtype=dtype, backend=backend, requires_grad=True)
        y = getattr(x, method)(dim=dim).log().sum()
        y.backward()

        tdtype = getattr(torch, dtype.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        ty = getattr(tx, method)(dim=dim).log().sum()
        ty.backward()
        
        self._check_tensors(ty, y, msg=f'{name}@forward')
        self._check_tensors(tx.grad, x.grad, msg=f'{name}@x_grad')

    def _check_tensors(self, a, b, tol=1e-5, msg=''):
        self.assertTrue(check_tensors(a.tolist(), b.tolist(), tol=tol, show_diff=True), msg=msg)

    def _op_to_method(self, op):
        return {
            OP.SUM_REDUCE: 'sum',
            OP.MEAN_REDUCE: 'mean',
        }[op]
