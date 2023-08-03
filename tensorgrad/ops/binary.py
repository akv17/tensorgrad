from .interface import Op
from ..const import OP


class Add(Op):
    NAME = OP.ADD

    def __init__(self, a, b):
        self.out = None
        self.a = a
        self.b = b

    def forward(self):
        self.out = self.a.zeros_like()
        self.out.data = self.a.data + self.b.data
        return self.out

    def backward(self):
        if self.a.requires_grad:
            self.a.grad += self.out.grad
        if self.b.requires_grad:
            self.b.grad += self.out.grad


class Sub(Op):
    NAME = OP.SUB

    def __init__(self, a, b):
        self.out = None
        self.a = a
        self.b = b

    def forward(self):
        self.out = self.a.zeros_like()
        self.out.data = self.a.data - self.b.data
        return self.out

    def backward(self):
        if self.a.requires_grad:
            self.a.grad += self.out.grad
        if self.b.requires_grad:
            self.b.grad += -1.0 * self.out.grad


class Mul(Op):
    NAME = OP.MUL

    def __init__(self, a, b):
        self.out = None
        self.a = a
        self.b = b

    def forward(self):
        self.out = self.a.zeros_like()
        self.out.data = self.a.data * self.b.data
        return self.out

    def backward(self):
        if self.a.requires_grad:
            self.a.grad += self.b.data * self.out.grad
        if self.b.requires_grad:
            self.b.grad += self.a.data * self.out.grad


class Div(Op):
    NAME = OP.DIV

    def __init__(self, a, b):
        self.out = None
        self.a = a
        self.b = b

    def forward(self):
        self.out = self.a.zeros_like()
        self.out.data = self.a.data / self.b.data
        return self.out

    def backward(self):
        if self.a.requires_grad:
            self.a.grad += 1.0 / self.b.data * self.out.grad
        if self.b.requires_grad:
            self.b.grad += (-self.a.data / (self.b.data ** 2)) * self.out.grad
