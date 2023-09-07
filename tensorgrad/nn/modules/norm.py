from abc import ABC, abstractclassmethod

from .base import Module, Parameter
from .. import init


class _GeneralizedNorm(Module, ABC):
    
    def __init__(self, num_features, eps=1e-05, dtype=None, device=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.dim = None
        self._calculate_dim()
        
        self.weight = Parameter.empty(self.num_features, dtype=dtype, device=device)
        self.bias = Parameter.empty(self.num_features, dtype=dtype, device=device)
        self.reset_parameters()

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
        return self
    
    def reset_parameters(self):
        init.ones(self.weight)
        init.zeros(self.bias)

    @abstractclassmethod
    def _calculate_dim(self): pass


class BatchNorm1d(_GeneralizedNorm):
    """
    check 2d only
    check batch size > 1
    accumulate running

    weight: [num_features,]
    bias: [num_features,]
    """
    
    def _calculate_dim(self):
        self.dim = 0


class BatchNorm2d(_GeneralizedNorm):
    """
    check 4d only
    check batch size > 1
    accumulate running

    weight: [num_features,]
    bias: [num_features,]
    """

    def forward(self, x):
        x = x.transpose(0, 2, 3, 1)
        x = super().forward(x)
        x = x.transpose(0, 3, 1, 2)
        return x
    
    def _calculate_dim(self):
        # B x H x W with channel-last.
        self.dim = (0, 1, 2)


class LayerNorm(_GeneralizedNorm):
    """
    """
    
    def __init__(self, normalized_shape, eps=1e-05, dtype=None, device=None):
        normalized_shape = (normalized_shape,) if not isinstance(normalized_shape, (tuple, list)) else normalized_shape
        self.normalized_shape = normalized_shape
        super().__init__(num_features=self.normalized_shape, eps=eps, dtype=dtype, device=device)

    def _calculate_dim(self):
        self.dim = tuple(range(-len(self.normalized_shape), 0))
