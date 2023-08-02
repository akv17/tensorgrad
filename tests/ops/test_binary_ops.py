import unittest
import numpy as np

from parameterized import parameterized

from tensorgrad.tensor import Tensor
from tensorgrad.const import DTYPE, BACKEND, OP
from ..util import require_torch, check_tensors

torch = require_torch()


ADD_CASES = [
    (OP.ADD, (1,), BACKEND.NUMPY, DTYPE.FLOAT32),
    (OP.ADD, (10,), BACKEND.NUMPY, DTYPE.FLOAT32),
    (OP.ADD, (100,), BACKEND.NUMPY, DTYPE.FLOAT32),
    (OP.ADD, (1000,), BACKEND.NUMPY, DTYPE.FLOAT32),
    (OP.ADD, (1,), BACKEND.NUMPY, DTYPE.FLOAT64),
    (OP.ADD, (10,), BACKEND.NUMPY, DTYPE.FLOAT64),
    (OP.ADD, (100,), BACKEND.NUMPY, DTYPE.FLOAT64),
    (OP.ADD, (1000,), BACKEND.NUMPY, DTYPE.FLOAT64),
]

MUL_CASES = [
    (OP.MUL, (1,), BACKEND.NUMPY, DTYPE.FLOAT32),
    (OP.MUL, (10,), BACKEND.NUMPY, DTYPE.FLOAT32),
    (OP.MUL, (100,), BACKEND.NUMPY, DTYPE.FLOAT32),
    (OP.MUL, (1000,), BACKEND.NUMPY, DTYPE.FLOAT32),
    (OP.MUL, (1,), BACKEND.NUMPY, DTYPE.FLOAT64),
    (OP.MUL, (10,), BACKEND.NUMPY, DTYPE.FLOAT64),
    (OP.MUL, (100,), BACKEND.NUMPY, DTYPE.FLOAT64),
    (OP.MUL, (1000,), BACKEND.NUMPY, DTYPE.FLOAT64),
]

CASES = ADD_CASES + MUL_CASES

class TestBinaryOps(unittest.TestCase):

    @parameterized.expand(CASES)
    def test(self, op, shape, backend, dtype):
        name = f'{op}@{shape}@{backend}@{dtype}'
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
        }[op]
