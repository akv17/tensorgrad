from .base import Module


class Sequential(Module):

    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self._modules[str(i)] = m

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())
    
    def __getitem__(self, ix):
        return self._modules[str(ix)]
    
    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x
