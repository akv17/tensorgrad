from .util.np import NumpyProvider
from ..stubs import BaseOp, UnaryOp
from ..dispatch import OpDispatch
from ...const import OP, DEVICE, DTYPE


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


@OpDispatch.register(OP.INVERT, DEVICE.CPU)
class Invert(UnaryOp):
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
