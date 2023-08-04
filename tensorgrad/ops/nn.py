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
        exp = self.x.data.exp()
        norm = exp.sum()
        self.exp = exp
        self.norm = norm
        out_data = exp / norm
        self.out = self.x.zeros(out_data.shape)
        self.out.data = out_data
        self.out.grad = out_data.zeros_like()
        return self.out

    def backward(self):
        if self.x.requires_grad:
            softmax = self.out.data
            # grad when i != j
            jacob = -(softmax.unsqueeze(1)).matmul(softmax.unsqueeze(0))
            # grad when i == j
            d_ii = softmax * (1.0 - softmax)
            jacob = jacob.fill_diagonal(d_ii)
            grad = jacob.matmul(self.out.grad)
            self.x.grad += grad
            
            # import numpy as np
            # sm = self.out.data.data
            # jb = -sm.reshape(-1, 1).dot(sm.reshape(1, -1))
            # a = sm * (1.0 - sm)
            # np.fill_diagonal(jb, a)
            # g = jb.dot(self.out.grad.data)
            # g = type(self.out.data)(np, g)
            # self.x.grad += g
