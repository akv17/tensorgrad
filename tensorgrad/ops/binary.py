from .interface import Op
from .util import accumulate_broadcasted_grad
from ..const import OP


class Add(Op):
    NAME = OP.ADD

    def __init__(self, a, b):
        self.out = None
        self.a = a
        self.b = b

    def forward(self):
        data = self.a.data + self.b.data
        self.out = self.a.from_data(data)
        return self.out

    def backward(self):
        if self.a.requires_grad:
            a_grad = self.out.grad
            a_grad = accumulate_broadcasted_grad(self.a, a_grad)
            self.a.grad += a_grad
        if self.b.requires_grad:
            b_grad = self.out.grad
            b_grad = accumulate_broadcasted_grad(self.b, b_grad)
            self.b.grad += b_grad


class Sub(Op):
    NAME = OP.SUB

    def __init__(self, a, b):
        self.out = None
        self.a = a
        self.b = b

    def forward(self):
        data = self.a.data - self.b.data
        self.out = self.a.from_data(data)
        return self.out

    def backward(self):
        if self.a.requires_grad:
            a_grad = self.out.grad
            a_grad = accumulate_broadcasted_grad(self.a, a_grad)
            self.a.grad += a_grad
        if self.b.requires_grad:
            b_grad = -1.0 * self.out.grad
            b_grad = accumulate_broadcasted_grad(self.b, b_grad)
            self.b.grad += b_grad


class Mul(Op):
    NAME = OP.MUL

    def __init__(self, a, b):
        self.out = None
        self.a = a
        self.b = b

    def forward(self):
        data = self.a.data * self.b.data
        self.out = self.a.from_data(data)
        return self.out

    def backward(self):
        if self.a.requires_grad:
            a_grad = self.b.data * self.out.grad
            a_grad = accumulate_broadcasted_grad(self.a, a_grad)
            self.a.grad += a_grad
        if self.b.requires_grad:
            b_grad = self.a.data * self.out.grad
            b_grad = accumulate_broadcasted_grad(self.b, b_grad)
            self.b.grad += b_grad


class Div(Op):
    NAME = OP.DIV

    def __init__(self, a, b):
        self.out = None
        self.a = a
        self.b = b

    def forward(self):
        data = self.a.data / self.b.data
        self.out = self.a.from_data(data)
        return self.out

    def backward(self):
        if self.a.requires_grad:
            a_grad = 1.0 / self.b.data * self.out.grad
            a_grad = accumulate_broadcasted_grad(self.a, a_grad)
            self.a.grad += a_grad
        if self.b.requires_grad:
            b_grad = (-self.a.data / (self.b.data ** 2)) * self.out.grad
            b_grad = accumulate_broadcasted_grad(self.b, b_grad)
            self.b.grad += b_grad
