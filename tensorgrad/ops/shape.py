from .interface import Op
from ..const import OP


class Squeeze(Op):
    NAME = OP.SQUEEZE

    def __init__(self, x, *, dim):
        self.out = None
        self.x = x
        self.dim = dim

    def forward(self):
        self.out = self.x.zeros_like()
        self.out.data = self.x.data.squeeze(self.dim)
        self.out.grad = self.out.grad.squeeze(self.dim)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.out.grad.unsqueeze(self.dim)


class Unsqueeze(Op):
    NAME = OP.UNSQUEEZE

    def __init__(self, x, *, dim):
        self.out = None
        self.x = x
        self.dim = dim

    def forward(self):
        self.out = self.x.zeros_like()
        self.out.data = self.x.data.unsqueeze(self.dim)
        self.out.grad = self.out.grad.unsqueeze(self.dim)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.out.grad.squeeze(self.dim)


class Reshape(Op):
    NAME = OP.RESHAPE

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


class Permute(Op):
    NAME = OP.PERMUTE

    def __init__(self, x, *, dims):
        self.out = None
        self.x = x
        self.dims = dims
        self.dims_grad = tuple(self.dims.index(i) for i in range(self.x.ndim))
    
    def forward(self):
        data = self.x.data.permute(self.dims)
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.out.grad.permute(self.dims_grad)


class Select(Op):
    NAME = OP.SELECT

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
            grad = self.x.data.zeros_like()
            grad = grad.put(self.slice_, self.out.grad)
            self.x.grad += grad
