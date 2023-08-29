import unittest

import numpy as np
from parameterized import parameterized

import tensorgrad
from tests.util import require_torch, check_tensors, generate_cases

torch = require_torch()

BACKENDS_TESTED = [tensorgrad.BACKEND.NUMPY]
DTYPES_TESTED = [tensorgrad.DTYPE.FLOAT32]


class TestOptim(unittest.TestCase):

    @parameterized.expand(
        generate_cases(
            [(4,), (128), (16, 32), (8, 16, 32)],
            [1, 4, 16],
            [0.1, 0.001, 1e-5],
            [None, 0.9],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
    )
    def test_sgd(self, shape, num_steps, lr, momentum, backend, dtype):
        name = f'{shape}::{num_steps}::{lr}::{momentum}::{backend}::{dtype}'

        _x = np.random.uniform(0.01, 1.0, size=shape)

        tdtype = getattr(torch, dtype.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        tx = torch.nn.Parameter(tx)
        toptim = torch.optim.SGD([tx], lr=lr, momentum=momentum or 0.0)

        x = tensorgrad.Tensor(tx.detach().numpy(), name='x', backend=backend, dtype=dtype, requires_grad=True)
        optim = tensorgrad.nn.SGD([x], lr=lr, momentum=momentum)

        for _ in range(num_steps):
            toptim.zero_grad()
            tf = tx.exp().mean()
            tf.backward()
            toptim.step()

            optim.zero_grad()
            f = x.exp().mean()
            f.backward()
            optim.step()
            
            self.assertTrue(check_tensors(tf, f, tol=1e-5, show_diff=True), msg=f'forward@{name}')
            self.assertTrue(check_tensors(tx.grad, x.grad, tol=1e-5, show_diff=True), msg=f'x_grad@{name}')

    @parameterized.expand(
        generate_cases(
            [(4,), (128), (16, 32), (8, 16, 32)],
            [1, 4, 16],
            [0.001],
            [(0.9, 0.999)],
            [1e-8],
            BACKENDS_TESTED,
            DTYPES_TESTED
        )
    )
    def test_adam(self, shape, num_steps, lr, betas, eps, backend, dtype):
        name = f'{shape}::{num_steps}::{lr}::{betas}::{eps}::{backend}::{dtype}'

        _x = np.random.uniform(0.01, 1.0, size=shape)

        tdtype = getattr(torch, dtype.value)
        tx = torch.tensor(_x, dtype=tdtype, requires_grad=True)
        tx = torch.nn.Parameter(tx)
        toptim = torch.optim.Adam([tx], lr=lr, betas=betas, eps=eps)

        x = tensorgrad.Tensor(tx.detach().numpy(), name='x', backend=backend, dtype=dtype, requires_grad=True)
        optim = tensorgrad.nn.Adam([x], lr=lr, betas=betas, eps=eps)

        for _ in range(num_steps):
            toptim.zero_grad()
            tf = tx.exp().mean()
            tf.backward()
            toptim.step()

            optim.zero_grad()
            f = x.exp().mean()
            f.backward()
            optim.step()
            
            self.assertTrue(check_tensors(tf, f, tol=1e-5, show_diff=True), msg=f'forward@{name}')
            self.assertTrue(check_tensors(tx.grad, x.grad, tol=1e-5, show_diff=True), msg=f'x_grad@{name}')
