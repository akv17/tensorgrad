from .base import NumpyOp
from .util.grad import accumulate_broadcasted_grad
from ..stubs import BinaryOp
from ...const import OP


class Add(BinaryOp, NumpyOp):
    _NAME = OP.ADD 

    def forward(self):
        o = self.a.data + self.b.data
        self.out = self.a.from_data(o)
        return self.out
    
    def backward(self):
        if self.a.requires_grad:
            a_grad = self.out.grad
            a_grad = accumulate_broadcasted_grad(self.np, self.a, a_grad)
            self.a.grad += a_grad
        if self.b.requires_grad:
            b_grad = self.out.grad
            b_grad = accumulate_broadcasted_grad(self.np, self.b, b_grad)
            self.b.grad += b_grad


class Sub(BinaryOp, NumpyOp):
    _NAME = OP.SUB

    def forward(self):
        data = self.a.data - self.b.data
        self.out = self.a.from_data(data)
        return self.out

    def backward(self):
        if self.a.requires_grad:
            a_grad = self.out.grad
            a_grad = accumulate_broadcasted_grad(self.np, self.a, a_grad)
            self.a.grad += a_grad
        if self.b.requires_grad:
            b_grad = -1.0 * self.out.grad
            b_grad = accumulate_broadcasted_grad(self.np, self.b, b_grad)
            self.b.grad += b_grad


class Mul(BinaryOp, NumpyOp):
    _NAME = OP.MUL

    def forward(self):
        data = self.a.data * self.b.data
        self.out = self.a.from_data(data)
        return self.out

    def backward(self):
        if self.a.requires_grad:
            a_grad = self.b.data * self.out.grad
            a_grad = accumulate_broadcasted_grad(self.np, self.a, a_grad)
            self.a.grad += a_grad
        if self.b.requires_grad:
            b_grad = self.a.data * self.out.grad
            b_grad = accumulate_broadcasted_grad(self.np, self.b, b_grad)
            self.b.grad += b_grad


class Div(BinaryOp, NumpyOp):
    _NAME = OP.DIV

    def forward(self):
        data = self.a.data / self.b.data
        self.out = self.a.from_data(data)
        return self.out

    def backward(self):
        if self.a.requires_grad:
            a_grad = 1.0 / self.b.data * self.out.grad
            a_grad = accumulate_broadcasted_grad(self.np, self.a, a_grad)
            self.a.grad += a_grad
        if self.b.requires_grad:
            b_grad = (-self.a.data / (self.b.data ** 2)) * self.out.grad
            b_grad = accumulate_broadcasted_grad(self.np, self.b, b_grad)
            self.b.grad += b_grad
