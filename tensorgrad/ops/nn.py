from .interface import Op
from ..const import OP


class Relu(Op):
    NAME = OP.RELU

    def __init__(self, x):
        self.out = None
        self.x = x
        self.mask = None

    def forward(self):
        self.out = self.x.copy()
        self.mask = self.x.data <= 0.0
        self.out.data = self.out.data.fill(self.mask, 0.0)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            mask = self.x.ones_like()
            mask.data = mask.data.fill(self.mask, 0.0)
            self.x.grad += mask.data * self.out.grad


class Sigmoid(Op):
    NAME = OP.SIGMOID

    def __init__(self, x):
        self.out = None
        self.x = x

    def forward(self):
        self.out = self.x.zeros_like()
        self.out.data = 1.0 / (1.0 + (-self.x.data).exp())
        return self.out

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.out.data * (1.0 - self.out.data) * self.out.grad
