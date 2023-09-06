from abc import ABC, abstractmethod

from ...tensor import Tensor


class Parameter(Tensor):

    def __repr__(self):
        value = super().__repr__()
        value = value.replace('Tensor', 'Parameter')
        return value


class Module(ABC):

    def __init__(self):
        self._parameters = []
        self._modules = []

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters.append(value)
        elif isinstance(value, Module):
            self._modules.append(value)
        super().__setattr__(name, value)

    def __call__(self, *args, **kwargs):
        self._check_input(*args, **kwargs)
        x = self.forward(*args, **kwargs)
        return x

    @abstractmethod
    def forward(self, *args, **kwargs): pass

    @abstractmethod
    def init_from_torch(self, module): pass

    def parameters(self):
        return self._parameters.copy()
    
    def _check_input(self, *args, **kwargs): pass
