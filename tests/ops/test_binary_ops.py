import unittest
import numpy as np

from parameterized import parameterized

from tensorgrad.tensor import Tensor
from tensorgrad.const import DTYPE, BACKEND, OP
from ..util import require_torch, check_tensors, generate_cases

torch = require_torch()

OPS_TESTED = (
    OP.ADD,
    OP.MUL,
    OP.SUB,
    OP.DIV,
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


class TestBinaryOps(unittest.TestCase):

    @parameterized.expand(CASES)
    def test(self, op, shape, backend, dtype):
        name = f'{op}::{shape}::{backend}::{dtype}'
        method = self._op_to_method(op)

        _a = np.random.random(shape).tolist()
        _b = np.random.random(shape).tolist()
        a = Tensor(_a, name='a', dtype=dtype, backend=backend, requires_grad=True)
        b = Tensor(_b, name='b', dtype=dtype, backend=backend, requires_grad=True)
        c = getattr(a, method)(b).sum()
        c.backward()

        tdtype = getattr(torch, dtype.value)
        ta = torch.tensor(_a, dtype=tdtype, requires_grad=True)
        tb = torch.tensor(_b, dtype=tdtype, requires_grad=True)
        tc = getattr(ta, method)(tb).sum()
        tc.backward()

        self._check_tensors(tc, c, msg=f'{name}@forward')
        self._check_tensors(ta.grad, a.grad, msg=f'{name}@a_grad')
        self._check_tensors(tb.grad, b.grad, msg=f'{name}@b_grad')
    
    def _check_tensors(self, a, b, tol=1e-5, msg=''):
        self.assertTrue(check_tensors(a.tolist(), b.tolist(), tol=tol), msg=msg)

    def _op_to_method(self, op):
        return {
            OP.ADD: '__add__',
            OP.MUL: '__mul__',
            OP.SUB: '__sub__',
            OP.DIV: '__truediv__',
        }[op]
