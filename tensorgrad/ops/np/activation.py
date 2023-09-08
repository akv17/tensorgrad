from .base import NumpyOp
from ..stubs import UnaryOp
from ...const import OP


class ReLU(UnaryOp, NumpyOp):
    _NAME = OP.RELU
    
    def forward(self):
        data = self.x.data.copy()
        self.mask = data <= 0.0
        data[self.mask] = 0.0
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            u = self.out.grad.copy()
            u[self.mask] = 0.0
            self.x.grad += u


class Sigmoid(UnaryOp, NumpyOp):
    _NAME = OP.SIGMOID

    def forward(self):
        data = 1.0 / (1.0 + self.np.exp(-self.x.data))
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.out.data * (1.0 - self.out.data) * self.out.grad
