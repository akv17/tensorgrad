from .base import NumpyOp
from .util.grad import accumulate_broadcasted_grad
from ..stubs import BinaryOp
from ...const import OP


class Matmul(BinaryOp, NumpyOp):
    _NAME = OP.MATMUL
    
    def forward(self):
        data = self.np.matmul(self.a.data, self.b.data)
        self.out = self.a.from_data(data)
        return self.out

    def backward(self):
        # notes on handling multidim tensors:
        # - no matter how many dims tensors have, this op may always be seen as collection of 2d matmuls.
        # - to leverage this we reshape all tensors to 3d shape [-1, prelast_dim, last_dim]
        # - then computing local grads is simple:
        #       downstream_grad_a = upstream_grad * b.T
        #       downstream_grad_b = a.T * upstream_grad
        # - finally we reshape all downstream grads back to match shape of the inputs.

        if self.a.requires_grad:
            u_grad = self.out.grad
            u_grad = u_grad.reshape(-1, u_grad.shape[-2], u_grad.shape[-1])
            b = self.b.data
            b = b.reshape(-1, b.shape[-2], b.shape[-1])
            b = self.np.transpose(b, ([0, 2, 1]))
            d_grad = self.np.matmul(u_grad, b)
            d_grad = accumulate_broadcasted_grad(self.np, self.a, d_grad)
            d_grad = d_grad.reshape(self.a.shape)
            self.a.grad += d_grad
        if self.b.requires_grad:
            u_grad = self.out.grad
            u_grad = u_grad.reshape(-1, u_grad.shape[-2], u_grad.shape[-1])
            a = self.a.data
            a = a.reshape(-1, a.shape[-2], a.shape[-1])
            a = self.np.transpose(a, ([0, 2, 1]))
            d_grad = self.np.matmul(a, u_grad)
            d_grad = accumulate_broadcasted_grad(self.np, self.b, d_grad)
            d_grad = d_grad.reshape(self.b.shape)
            self.b.grad += d_grad
