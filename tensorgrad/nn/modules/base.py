from abc import ABC, abstractmethod

from ...tensor import Tensor


class Parameter(Tensor):

    def __repr__(self):
        value = super().__repr__()
        value = value.replace('Tensor', 'Parameter')
        return value
    

class Buffer(Tensor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad = False

    def __repr__(self):
        value = super().__repr__()
        value = value.replace('Tensor', 'Buffer')
        return value


class Module(ABC):

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self.training = False

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Buffer):
            self._buffers[name] = value
        super().__setattr__(name, value)

    def __call__(self, *args, **kwargs):
        self._check_input(*args, **kwargs)
        x = self.forward(*args, **kwargs)
        return x

    @abstractmethod
    def forward(self, *args, **kwargs): pass

    def to(self, device):
        params_and_buffers = [*self._parameters.values(), *self._buffers.values()]
        for v in params_and_buffers:
            # must move inplace to keep `id(v)` the same.
            # for example to maintain reference by optimizers when passing params to optimizers.
            v.to(device, inplace=True)
        for m in self._modules.values():
            m.to(device)
    
    def cpu(self):
        self.to('cpu')
    
    def cuda(self):
        self.to('cuda')

    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()
    
    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()

    def init_from_torch(self, module):
        pass
    
    def modules(self):
        return list(self.named_modules().values())
    
    def parameters(self):
        return list(self.named_parameters().values())
    
    def buffers(self):
        return list(self.named_buffers().values())
    
    def named_modules(self):
        # prefix 's' stands for 'sub' -> 'submodule'.
        kv = {}
        for mn, m in self._modules.items():
            kv[mn] = m
            for smn, sm in m.named_modules().items():
                smk = f'{mn}.{smn}'
                kv[smk] = sm
        return kv

    def named_parameters(self):
        kv = self._named_params_and_buffers()['params']
        return kv
    
    def named_buffers(self):
        kv = self._named_params_and_buffers()['buffers']
        return kv

    def _named_params_and_buffers(self):
        kv = {'params': self._parameters.copy(), 'buffers': self._buffers.copy()}
        for mn, m in self._modules.items():
            for pn, p in m.named_parameters().items():
                pk = f'{mn}.{pn}'
                kv['params'][pk] = p
            for bn, b in m.named_buffers().items():
                bk = f'{mn}.{bn}'
                kv['buffers'][bk] = b
        return kv
    
    def _check_input(self, *args, **kwargs): pass
