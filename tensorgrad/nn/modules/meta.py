from .base import Module


class Sequential(Module):
    """
    Applies a sequence of modules in order one after another.  

    **Parameters**:  
    - `modules: Iterable[Module]:` iterable of modules to apply  

    **Input:** `(*)`  
    **Output:** `(*)`  
    **Weights:** `None`  
    """

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
    
    def forward(self, x, **kwargs):
        for m in self._modules.values():
            x = m(x, **kwargs)
        return x
