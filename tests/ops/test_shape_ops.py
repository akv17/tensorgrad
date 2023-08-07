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
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [1],
            [
                (10, 1),
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [0],
            [
                (1, 10, 10),
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [1],
            [
                (10, 1, 10),
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [2],
            [
                (10, 10, 1),
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
    )
    def test_squeeze(self, dim, shape, backend, dtype):
        self._test(shape=shape, op=OP.SQUEEZE, backend=backend, dtype=dtype, kwargs={'dim': dim})

    @parameterized.expand(
        generate_cases(
            [0, 1],
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
            [0, 1, 2],
            [
                (5, 10),
                (50, 100),
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [0, 1, 2, 3],
            [
                (2, 5, 10),
                (5, 50, 100),
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
    )
    def test_unsqueeze(self, dim, shape, backend, dtype):
        self._test(shape=shape, op=OP.UNSQUEEZE, backend=backend, dtype=dtype, kwargs={'dim': dim})

    @parameterized.expand(
        generate_cases(
            [
                [(6,), (2, 3)],
                [(3, 1), (1, 3)],
                [(3, 4), (2, 6)],
                [(2, 3, 4), (2, 12)],
                [(2, 12), (2, 3, 4)],
                [(2, 3, 4, 5), (6, 4, 5)],
                [(2, 3, 4, 5), (-1, 5)],
                [(2, 3, 4, 5), (24, -1)],
                [(2, 3, 4, 5), (-1,)],
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
    )
    def test_reshape(self, shapes, backend, dtype):
        shape_in, shape_out = shapes
        self._test(shape=shape_in, op=OP.RESHAPE, backend=backend, dtype=dtype, args=(shape_out,))
    
    @parameterized.expand(
        generate_cases(
            [
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
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
    )
    def test_permute(self, shapes, backend, dtype):
        shape_in, dims = shapes
        self._test(shape=shape_in, op=OP.PERMUTE, backend=backend, dtype=dtype, args=(dims,))

    def _test(self, shape, op, backend, dtype, args=None, kwargs=None):
        args = args or ()
        kwargs = kwargs or {}
        name = f'{op}::{shape}::{backend}::{dtype}::{args}::{kwargs}'
        method = self._op_to_method(op)

        _x = np.random.random(shape).tolist()

        x = Tensor(_x, name='x', dtype=dtype, backend=backend, requires_grad=True)
        y = getattr(x, method)(*args, **kwargs).log().sum()
        y.backward()

        tdtype = getattr(torch, dtype.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        ty = getattr(tx, method)(*args, **kwargs).log().sum()
        ty.backward()

        self._check_tensors(ty, y, msg=f'{name}@forward')
        self._check_tensors(tx.grad, x.grad, msg=f'{name}@x_grad')

    def _check_tensors(self, a, b, tol=1e-5, msg=''):
        self.assertTrue(check_tensors(a.tolist(), b.tolist(), tol=tol, show_diff=True), msg=msg)

    def _op_to_method(self, op):
        return {
            OP.SQUEEZE: 'squeeze',
            OP.UNSQUEEZE: 'unsqueeze',
            OP.RESHAPE: 'reshape',
            OP.PERMUTE: 'permute',
        }[op]
