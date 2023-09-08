from .base import NumpyOp
from ..stubs import BaseOp
from ...const import OP


class Squeeze(BaseOp, NumpyOp):
    _NAME = OP.SQUEEZE

    def __init__(self, x, *, dim):
        self.out = None
        self.x = x
        self.dim = dim

    def forward(self):
        data = self.np.squeeze(self.x.data, self.dim)
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.np.expand_dims(self.out.grad, self.dim)


class Unsqueeze(BaseOp, NumpyOp):
    _NAME = OP.UNSQUEEZE
    
    def __init__(self, x, *, dim):
        self.out = None
        self.x = x
        self.dim = dim

    def forward(self):
        data = self.np.expand_dims(self.x.data, self.dim)
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.np.squeeze(self.out.grad, self.dim)


class Reshape(BaseOp, NumpyOp):
    _NAME = OP.RESHAPE

    def __init__(self, x, *, shape):
        self.out = None
        self.x = x
        self.shape = shape

    def forward(self):
        data = self.x.data.reshape(self.shape)
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.out.grad.reshape(self.x.grad.shape)


class Permute(BaseOp, NumpyOp):
    _NAME = OP.PERMUTE

    def __init__(self, x, *, dims):
        self.out = None
        self.x = x
        self.dims = dims
        self.dims_grad = tuple(self.dims.index(i) for i in range(self.x.ndim))
    
    def forward(self):
        data = self.np.transpose(self.x.data, self.dims)
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.np.transpose(self.out.grad, self.dims_grad)


class Concat(BaseOp, NumpyOp):
    _NAME = OP.CONCAT

    def __init__(self, x, *, dim):
        self.x = x
        self.dim = dim
    
    def forward(self):
        data = [xi.data for xi in self.x]
        x = self.np.concatenate(data, axis=self.dim)
        head = self.x[0]
        self.out = head.from_data(x)
        return self.out
    
    def backward(self):
        sections = []
        idx = 0
        for xi in self.x[:-1]:
            idx += xi.shape[self.dim]
            sections.append(idx)

        g = self.np.split(self.out.grad, sections, axis=self.dim)
        for xi, gi in zip(self.x, g):
            if xi.requires_grad:
                xi.grad += gi
