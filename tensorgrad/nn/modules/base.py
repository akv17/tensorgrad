from abc import ABC, abstractmethod

from ...tensor import Tensor


class Parameter(Tensor):

    def __repr__(self):
        value = super().__repr__()
        value = value.replace('Tensor', 'Parameter')
        return value


class Module(ABC):

    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def __call__(self, *args, **kwargs):
        self._check_input(*args, **kwargs)
        x = self.forward(*args, **kwargs)
        return x

    @abstractmethod
    def forward(self, *args, **kwargs): pass

    def init_from_torch(self, module):
        pass

    def parameters(self):
        return list(self.named_parameters().values())
    
    def modules(self):
        return list(self.named_modules().values())

    def named_parameters(self):
        kv = {**self._parameters}
        for mn, m in self._modules.items():
            for pn, p in m.named_parameters().items():
                pk = f'{mn}.{pn}'
                kv[pk] = p
        return kv

    def named_modules(self):
        # prefix 's' stands for 'sub' -> 'submodule'.
        kv = {}
        for mn, m in self._modules.items():
            kv[mn] = m
            for smn, sm in m.named_modules().items():
                smk = f'{mn}.{smn}'
                kv[smk] = sm
        return kv
        
    def _check_input(self, *args, **kwargs): pass
