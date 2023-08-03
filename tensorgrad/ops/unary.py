from .interface import Op
from ..const import OP


class Pow(Op):
    NAME = OP.POW
    
    def __init__(self, x, *, value):
        self.out = None
        self.x = x
        self.value = value
    
    def forward(self):
        self.out = self.x.zeros_like()
        self.out.data = self.x.data ** self.value
        return self.out
    
    def backward(self):
        if self.x.requires_grad:
            n = self.value
            self.x.grad += (n * self.x.data ** (n-1)) * self.out.grad


class Exp(Op):
    NAME = OP.EXP
    
    def __init__(self, x):
        self.out = None
        self.x = x
    
    def forward(self):
        self.out = self.x.zeros_like()
        self.out.data = self.x.data.exp()
        return self.out
    
    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.out.data * self.out.grad


class Log(Op):
    NAME = OP.LOG
    
    def __init__(self, x):
        self.out = None
        self.x = x
    
    def forward(self):
        self.out = self.x.zeros_like()
        self.out.data = self.x.data.log()
        return self.out
    
    def backward(self):
        if self.x.requires_grad:
            self.x.grad += 1.0 / self.x.data * self.out.grad
