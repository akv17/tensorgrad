from .base import Module, Parameter
from .. import init


class BatchNorm1d(Module):
    """
    check 2d only
    check batch size > 1
    accumulate running

    weight: [num_features,]
    bias: [num_features,]
    """
    
    def __init__(self, num_features, eps=1e-05, dtype=None, device=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.dim = 0
        self.weight = Parameter.empty((self.num_features,), dtype=dtype, device=device)
        self.bias = Parameter.empty((self.num_features,), dtype=dtype, device=device)

    def forward(self, x):
        mean = x.mean(self.dim, keepdim=True)
        std = ((x - mean) ** 2).mean(self.dim, keepdim=True)
        x_norm = (x - mean) / ((std + self.eps).sqrt())
        x = self.weight * x_norm + self.bias
        return x

    def init_from_torch(self, module):
        self.weight = Parameter(
            module.weight.detach().cpu().numpy(),
            dtype=self.weight.dtype,
            device=self.weight.device
        )
        self.bias = Parameter(
            module.bias.detach().cpu().numpy(),
            dtype=self.bias.dtype,
            device=self.bias.device
        )
    
    def reset_parameters(self):
        init.ones(self.weight)
        init.zeros(self.bias)


class BatchNorm2d(Module):
    """
    check 4d only
    check batch size > 1
    accumulate running

    weight: [num_features,]
    bias: [num_features,]
    """
    
    def __init__(self, num_features, eps=1e-05, dtype=None, device=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.dim = (0, 1, 2)
        self.weight = Parameter.empty((self.num_features,), dtype=dtype, device=device)
        self.bias = Parameter.empty((self.num_features,), dtype=dtype, device=device)

    def forward(self, x):
        x = x.transpose(0, 2, 3, 1)
        mean = x.mean(self.dim, keepdim=True)
        std = ((x - mean) ** 2).mean(self.dim, keepdim=True)
        x_norm = (x - mean) / ((std + self.eps).sqrt())
        x = self.weight * x_norm + self.bias
        x = x.transpose(0, 3, 1, 2)
        return x

    def init_from_torch(self, module):
        self.weight = Parameter(
            module.weight.detach().cpu().numpy(),
            dtype=self.weight.dtype,
            device=self.weight.device
        )
        self.bias = Parameter(
            module.bias.detach().cpu().numpy(),
            dtype=self.bias.dtype,
            device=self.bias.device
        )
    
    def reset_parameters(self):
        init.ones(self.weight)
        init.zeros(self.bias)


class LayerNorm(Module):
    """
    """
    
    def __init__(self, normalized_shape, eps=1e-05, dtype=None, device=None):
        super().__init__()
        normalized_shape = (normalized_shape,) if not isinstance(normalized_shape, (tuple, list)) else normalized_shape
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.dim = tuple(range(-len(normalized_shape), 0))
        self.weight = Parameter.empty(self.normalized_shape, dtype=dtype, device=device)
        self.bias = Parameter.empty(self.normalized_shape, dtype=dtype, device=device)

    def forward(self, x):
        mean = x.mean(self.dim, keepdim=True)
        std = ((x - mean) ** 2).mean(self.dim, keepdim=True)
        x_norm = (x - mean) / ((std + self.eps).sqrt())
        x = self.weight * x_norm + self.bias
        return x

    def init_from_torch(self, module):
        self.weight = Parameter(
            module.weight.detach().cpu().numpy(),
            dtype=self.weight.dtype,
            device=self.weight.device
        )
        self.bias = Parameter(
            module.bias.detach().cpu().numpy(),
            dtype=self.bias.dtype,
            device=self.bias.device
        )
    
    def reset_parameters(self):
        init.ones(self.weight)
        init.zeros(self.bias)