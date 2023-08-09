import math

from .interface import Module
from ..tensor import Tensor
from ..backend import BackendDispatch


class Parameter:

    def __init__(self, tensor):
        self.tensor = tensor

    def __repr__(self):
        return f'Parameter<{self.tensor}>'

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def update_(self, data):
        self.tensor.data += data

    def zero_grad_(self):
        self.tensor.grad = self.tensor.grad.zeros_like()


class Linear(Module):
    
    def __init__(self, in_features, out_features, bias=True, name=None, backend=None, dtype=None):
        self.in_features = in_features
        self.out_features = out_features
        self.with_bias = bias
        self.backend = backend
        self.dtype = dtype
        self.name = name or f'Linear@{id(self)}'

        self._backend = BackendDispatch.get(backend)

        self.weight = None
        self.bias = None
        self.initialize()

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        return out

    def forward(self, x):
        if x.ndim < 2:
            msg = f'Linear requires at least 2d input, got ndim={x.ndim}'
            raise Exception(msg)
        if self.with_bias:
            out = x.matmul(self.weight.transpose(1, 0)) + self.bias
        else:
            out = x.matmul(self.weight.transpose(1, 0))
        return out

    def initialize(self, weight=None, bias=None):
        if weight is None:
            k = math.sqrt(self.in_features)
            w = self._backend.random_uniform(-k, k, shape=(self.out_features, self.in_features))
        else:
            w = weight
        weight = Tensor(w, name=f'weight@{self.name}', backend=self.backend, dtype=self.dtype)
        self.weight = weight
        if self.with_bias:
            if bias is None:
                k = math.sqrt(self.in_features)
                b = self._backend.random_uniform(-k, k, shape=(self.out_features,))
            else:
                b = bias
            bias = Tensor(b, name=f'bias@{self.name}', backend=self.backend, dtype=self.dtype)
            self.bias = bias
        return self

    def parameters(self):
        return [self.weight, self.bias] if self.with_bias else [self.weight]


class ReLU(Module):

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        return out
    
    def forward(self, x):
        x = x.relu()
        return x
    
    def initialize(self):
        pass
    
    def parameters(self):
        return []


class Identity(Module):

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        return out
    
    def forward(self, x):
        return x
    
    def initialize(self):
        pass
    
    def parameters(self):
        return []


class Sigmoid(Module):

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        return out
    
    def forward(self, x):
        x = x.sigmoid()
        return x
    
    def initialize(self):
        pass
    
    def parameters(self):
        return []


class Sequential(Module):

    def __init__(self, *modules):
        self.modules = tuple(modules)

    def __len__(self):
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)
    
    def __getitem__(self, ix):
        return self.modules[ix]

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        return out
    
    def forward(self, x):
        for mod in self.modules:
            x = mod(x)
        return x
    
    def initialize(self):
        pass

    def parameters(self):
        return [p for m in self.modules for p in m.parameters()]
    