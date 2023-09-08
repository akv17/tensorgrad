from .base import NumpyOp
from ..stubs import BaseOp, UnaryOp
from ...const import OP, DTYPE


class Pow(BaseOp, NumpyOp):
    _NAME = OP.POW
    
    def __init__(self, x, *, value):
        self.out = None
        self.x = x
        self.value = value
    
    def forward(self):
        data = self.x.data ** self.value
        self.out = self.x.from_data(data)
        return self.out
    
    def backward(self):
        if self.x.requires_grad:
            n = self.value
            self.x.grad += (n * self.x.data ** (n-1)) * self.out.grad


class Exp(UnaryOp, NumpyOp):
    _NAME = OP.EXP
    
    def forward(self):
        data = self.np.exp(self.x.data)
        self.out = self.x.from_data(data)
        return self.out
    
    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.out.data * self.out.grad


class Log(UnaryOp, NumpyOp):
    _NAME = OP.LOG
    
    def forward(self):
        data = self.np.log(self.x.data)
        self.out = self.x.from_data(data)
        return self.out
    
    def backward(self):
        if self.x.requires_grad:
            self.x.grad += 1.0 / self.x.data * self.out.grad


class Invert(UnaryOp, NumpyOp):
    _NAME = OP.INVERT
    # tested via `nn.Dropout`
    
    def forward(self):
        x = self.x
        if x.dtype is not DTYPE.BOOL:
            msg = f'invert may only be called on boolean tensor, but got tensor of type: {x.dtype}'
            raise Exception(msg)
        data = ~x.data
        self.out = self.x.from_data(data)
        return self.out
    
    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.out.grad
