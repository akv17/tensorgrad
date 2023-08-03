from .interface import Op
from ..const import OP


class Relu(Op):
    NAME = OP.RELU

    def __init__(self, out, x):
        self.out = out
        self.x = x
        self.mask = None

    def forward(self):
        self.mask = self.x.data <= 0.0
        self.out.data = self.x.data.copy()
        self.out.data = self.out.data.fill(self.mask, 0.0)

    def backward(self):
        if self.x.requires_grad:
            mask = self.x.ones_like()
            mask.data = mask.data.fill(self.mask, 0.0)
            self.x.grad += mask.data * self.out.grad
