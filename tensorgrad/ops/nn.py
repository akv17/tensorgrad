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


class Softmax(Op):
    NAME = OP.SOFTMAX

    def __init__(self, x, *, dim):
        self.out = None
        self.x = x
        self.dim = dim

    def forward(self):
        # exp = self.x.data.exp()
        # norm = exp.sum(self.dim).unsqueeze(self.dim)
        # out = exp / norm
        # self.out = self.x.zeros_like()
        # self.out.data = out
        # return self.out
        x = self.x.detach()
        exp = x.exp(self.dim)
        norm = exp.sum(self.dim).unsqueeze(self.dim)
        self.out = exp / norm
        return self.out

    def backward(self):
        self.out.backward()
        
        # if self.x.requires_grad:
        #     pass
