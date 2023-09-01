from .interface import Module


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
    """
    weight: [out, in]
    bias: [out,]
    """
    
    def __init__(self, in_features, out_features, bias=True, name=None):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.name = name or f'Linear@{id(self)}'

        self.weight = None
        self.bias = None

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        return out

    def forward(self, x):
        w = self.weight.transpose(1, 0)
        b = self.bias if self.bias is not None else None
        out = x.matmul(w) + b if b is not None else x.matmul(w)
        return out

    def parameters(self):
        return [self.weight, self.bias] if self.use_bias else [self.weight]


class BatchNorm1d(Module):
    """
    check 2d only
    check batch size > 1
    accumulate running

    weight: [num_features,]
    bias: [num_features,]
    """
    
    def __init__(self, num_features, eps=1e-05, name=None):
        self.num_features = num_features
        self.eps = eps
        self.name = name or f'BatchNorm1D@{id(self)}'

        self.dim = 0
        self.weight = None
        self.bias = None

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        return out

    def forward(self, x):
        mean = x.mean(self.dim).unsqueeze(self.dim)
        std = ((x - mean) ** 2).mean(self.dim).unsqueeze(self.dim)
        x_norm = (x - mean) / ((std + self.eps).sqrt())
        x = self.weight * x_norm + self.bias
        return x

    def parameters(self):
        return [self.weight, self.bias] if self.use_bias else [self.weight]


class BatchNorm2d(Module):
    """
    check 4d only
    check batch size > 1
    accumulate running

    weight: [num_features,]
    bias: [num_features,]
    """
    
    def __init__(self, num_features, eps=1e-05, name=None):
        self.num_features = num_features
        self.eps = eps
        self.name = name or f'BatchNorm2D@{id(self)}'

        self.dim = 0
        self.weight = None
        self.bias = None

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        return out

    def forward(self, x):
        x = x.transpose(0, 2, 3, 1)
        x_flat = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3])
        mean = x_flat.mean(self.dim).unsqueeze(self.dim)
        std = ((x_flat - mean) ** 2).mean(self.dim).unsqueeze(self.dim)
        x_norm = (x_flat - mean) / ((std + self.eps).sqrt())
        x_flat = self.weight * x_norm + self.bias
        x = x_flat.reshape(x.shape)
        x = x.transpose(0, 3, 1, 2)
        return x

    def parameters(self):
        return [self.weight, self.bias] if self.use_bias else [self.weight]


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
    