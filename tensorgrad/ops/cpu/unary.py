from .util.np import NumpyProvider
from ..stubs import BaseOp, UnaryOp
from ..dispatch import OpDispatch
from ...const import OP, DEVICE


@OpDispatch.register(OP.POW, DEVICE.CPU)
class Pow(BaseOp):
    
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


@OpDispatch.register(OP.EXP, DEVICE.CPU)
class Exp(UnaryOp, NumpyProvider):
    
    def forward(self):
        data = self.np.exp(self.x.data)
        self.out = self.x.from_data(data)
        return self.out
    
    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.out.data * self.out.grad


@OpDispatch.register(OP.LOG, DEVICE.CPU)
class Log(UnaryOp, NumpyProvider):
    
    def forward(self):
        data = self.np.log(self.x.data)
        self.out = self.x.from_data(data)
        return self.out
    
    def backward(self):
        if self.x.requires_grad:
            self.x.grad += 1.0 / self.x.data * self.out.grad


@OpDispatch.register(OP.MASKED_FILL, DEVICE.CPU)
class MaskedFill(BaseOp):

    def __init__(self, x, *, mask, value):
        self.x = x
        self.mask = mask
        self.value = value

    def forward(self):
        data = self.x.data
        data[self.mask.data] = self.value
        self.x.data = data
        self.out = self.x
        return self.out
    
    def backward(self):
        # this op does not have a gradient.
        pass
