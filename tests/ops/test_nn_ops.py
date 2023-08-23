import unittest
import numpy as np

from parameterized import parameterized

from tensorgrad.tensor import Tensor
from tensorgrad.const import DTYPE, BACKEND, OP
from tests.util import require_torch, check_tensors, generate_cases

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


class TestNNOps(unittest.TestCase):

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
            [0, -1],
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
            [0, -1],
            [
                (2, 3),
                (200, 300),
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [0, 1, 2, -1],
            [
                (2, 3, 4),
                (20, 30, 40),
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
        +
        generate_cases(
            [0, 1, 2, 3, -1],
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

    @parameterized.expand(
        generate_cases(
            [
                [(2, 4), (4, 2)],
                [(2, 2), (2, 2)],
                [(4, 1), (1, 4)],
                [(1, 4), (4, 1)],
                [(128, 256), (256, 128)],
                [(2, 3, 4), (2, 4, 3)],
                [(20, 30, 40), (20, 40, 30)],
                [(2, 3, 4, 5), (2, 3, 5, 4)],
                [(20, 30, 40, 50), (20, 30, 50, 40)],
                [(2, 3, 4), (4, 5)],
                [(2, 5, 3, 4), (4, 5)],
                [(3, 2, 5, 3, 4), (4, 5)],
                [(4, 5), (2, 5, 4)],
                [(4, 5), (2, 3, 5, 4)],
                [(4, 5), (3, 2, 4, 5, 4)],
            ],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
    )
    def test_matmul(self, shapes, backend, dtype):
        name = f'matmul::{shapes}::{backend}::{dtype}'
        
        a_shape, b_shape = shapes
        _a = np.random.uniform(-1.0, 1.0, size=a_shape).tolist()
        _b = np.random.uniform(-1.0, 1.0, size=b_shape).tolist()

        a = Tensor(_a, name='a', dtype=dtype, backend=backend, requires_grad=True)
        b = Tensor(_b, name='b', dtype=dtype, backend=backend, requires_grad=True)
        c = a.matmul(b).sum()
        c.backward()

        tdtype = getattr(torch, dtype.value)
        ta = torch.tensor(_a, dtype=tdtype, requires_grad=True)
        tb = torch.tensor(_b, dtype=tdtype, requires_grad=True)
        tc = ta.matmul(tb).sum()
        tc.backward()

        self._check_tensors(tc, c, msg=f'{name}@forward')
        self._check_tensors(ta.grad, a.grad, msg=f'{name}@a_grad')
        self._check_tensors(tb.grad, b.grad, msg=f'{name}@b_grad')

    @parameterized.expand(
        generate_cases(
            [7],
            [6],
            [5],
            [(4, 4)],
            [(3, 3)],
            [False],
            [(1, 1)],
            [(0, 0)],
            BACKENDS_TESTED,
            DTYPES_TESTED,
        )
    )
    def test_conv2d(self, batch_size, in_channels, out_channels, input_size, kernel_size, bias, stride, padding, backend, dtype):
        input_shape = (batch_size, in_channels, *input_size)
        kernel_shape = (out_channels, in_channels, *kernel_size)
        name = f'{input_shape}::{kernel_shape}::{stride}::{padding}::{backend}::{dtype}'

        tdtype = getattr(torch, dtype.value)
        tx = torch.rand(input_shape, dtype=tdtype, requires_grad=True)
        tk = torch.rand(kernel_shape, dtype=tdtype, requires_grad=True)
        tb = torch.rand((out_channels,), dtype=tdtype, requires_grad=True) if bias else None
        to = torch.nn.functional.conv2d(tx, tk, tb, stride=stride, padding=padding).log().sum()
        to.backward()

        x = Tensor(tx.tolist(), name='x', backend=backend, dtype=dtype, requires_grad=True)
        k = Tensor(tk.tolist(), name='k', backend=backend, dtype=dtype, requires_grad=True)
        b = Tensor(tb.tolist(), name='b', backend=backend, dtype=dtype, requires_grad=False) if bias else None
        o = x.conv2d(k, bias=b, stride=stride, padding=padding).log().sum()
        o.backward()

        self._check_tensors(to, o, msg=f'{name}@forward')
        # self._check_tensors(tk.grad, k.grad, msg=f'{name}@k_grad')
        # self._check_tensors(tx.grad, x.grad, msg=f'{name}@x_grad')
        # if bias:
        #     self._check_tensors(tb.grad, b.grad, msg=f'{name}@b_grad')

    def _check_tensors(self, a, b, tol=1e-5, msg=''):
        self.assertTrue(check_tensors(a, b, tol=tol, show_diff=True), msg=msg)

    def _op_to_method(self, op):
        return {
            OP.RELU: 'relu',
            OP.SIGMOID: 'sigmoid',
        }[op]
