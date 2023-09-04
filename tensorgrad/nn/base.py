from abc import ABC, abstractmethod
from typing import Any

from ..tensor import Tensor


class Parameter(Tensor):

    def __repr__(self):
        value = super().__repr__()
        value = value.replace('Tensor', 'Parameter')
        return value


class ParameterSpec:

    def __init__(self, shape, dtype, device):
        self.shape = shape
        self.dtype = dtype
        self.device = device


class Module:

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
        x = self.forward(*args, **kwargs)
        return x

    @abstractmethod
    def forward(self, *args, **kwargs): pass

    @abstractmethod
    def init_from_torch(self, module): pass

    def parameters(self):
        return self._parameters.copy()
