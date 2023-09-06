from .base import Module


class Sequential(Module):

    def __init__(self, *modules):
        self.modules = tuple(modules)

    def __len__(self):
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)
    
    def __getitem__(self, ix):
        return self.modules[ix]
    
    def forward(self, x):
        for mod in self.modules:
            x = mod(x)
        return x
