from .base import NumpyOp
from ..stubs import BaseOp
from ...const import OP


class Select(BaseOp, NumpyOp):
    _NAME = OP.SELECT

    def __init__(self, x, *, slice_):
        self.out = None
        self.x = x
        self.slice_ = slice_
    
    def forward(self):
        data = self.x.data[self.slice_]
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            grad = self.np.zeros_like(self.x.data)
            grad[self.slice_] = self.out.grad
            self.x.grad += grad


class MaskedFill(BaseOp, NumpyOp):
    # this op does not have a gradient.

    _NAME = OP.MASKED_FILL

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
        pass


class Lookup(BaseOp, NumpyOp):
    _NAME = OP.LOOKUP

    # this is essentially an embedding implementation.
    # it's not possible to reuse Select for this purpose because embedding weight gradients must be accumulated.

    # tested via `nn.Embedding`

    def __init__(self, x, *, mask):
        self.out = None
        self.x = x
        self.mask = mask
    
    def forward(self):
        mask = self.mask.data
        mask = mask.ravel()
        data = self.x.data[mask]
        data = data.reshape(*self.mask.shape, data.shape[-1])
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            x = self.x.data
            g = self.np.zeros_like(x)
            u = self.out.grad.reshape(-1, x.shape[-1])
            m = self.mask.data.ravel()
            # this thing below computes `g[m[i]] += u[i]`.
            # basically this is accumulation by index.
            self.np.add.at(g, m, u)
            self.x.grad += g
