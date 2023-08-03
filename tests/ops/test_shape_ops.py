import unittest
import numpy as np

from parameterized import parameterized

from tensorgrad.tensor import Tensor
from tensorgrad.const import DTYPE, BACKEND, OP
from ..util import require_torch, check_tensors, generate_cases

torch = require_torch()

BACKENDS_TESTED = (
    BACKEND.NUMPY,
)
DTYPES_TESTED = (
    DTYPE.FLOAT32,
    DTYPE.FLOAT64,
)


class TestShapeOps(unittest.TestCase):

    @parameterized.expand(
        generate_cases(
            [0],
            [
                (1, 10),
            ],
            [OP.SQUEEZE],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [1],
            [
                (10, 1),
            ],
            [OP.SQUEEZE],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [0],
            [
                (1, 10, 10),
            ],
            [OP.SQUEEZE],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [1],
            [
                (10, 1, 10),
            ],
            [OP.SQUEEZE],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [2],
            [
                (10, 10, 1),
            ],
            [OP.SQUEEZE],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
    )
    def test_squeeze(self, dim, shape, op, backend, dtype):
        self._test(dim=dim, shape=shape, op=op, backend=backend, dtype=dtype)

    @parameterized.expand(
        generate_cases(
            [0, 1],
            [
                (1,),
                (10,),
                (100,),
                (1000,),
            ],
            [OP.UNSQUEEZE],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [0, 1, 2],
            [
                (10, 10),
                (100, 100),
            ],
            [OP.UNSQUEEZE],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [0, 1, 2, 3],
            [
                (10, 10, 10),
                (10, 100, 100),
            ],
            [OP.UNSQUEEZE],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
    )
    def test_unsqueeze(self, dim, shape, op, backend, dtype):
        self._test(dim=dim, shape=shape, op=op, backend=backend, dtype=dtype)

    def _test(self, dim, shape, op, backend, dtype):
        name = f'{op}::{shape}::{dim}::{backend}::{dtype}'
        method = self._op_to_method(op)

        _x = np.random.random(shape).tolist()

        x = Tensor(_x, name='x', dtype=dtype, backend=backend, requires_grad=True)
        y = getattr(x, method)(dim=dim).exp().sum()
        y.backward()

        tdtype = getattr(torch, dtype.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        ty = getattr(tx, method)(dim=dim).exp().sum()
        ty.backward()

        self._check_tensors(ty, y, msg=f'{name}@forward')
        self._check_tensors(tx.grad, x.grad, msg=f'{name}@x_grad')

    def _check_tensors(self, a, b, tol=1e-5, msg=''):
        self.assertTrue(check_tensors(a.tolist(), b.tolist(), tol=tol, show_diff=True), msg=msg)

    def _op_to_method(self, op):
        return {
            OP.SQUEEZE: 'squeeze',
            OP.UNSQUEEZE: 'unsqueeze',
        }[op]
