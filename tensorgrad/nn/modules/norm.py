from abc import ABC, abstractmethod

from .base import Module, Parameter, Buffer
from .. import init
from ...ctx import no_grad


class _BatchNorm(Module, ABC):
    
    def __init__(self, num_features, eps=1e-05, momentum=0.1, track_running_stats=True, dtype=None, device=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        self.dim = None
        self._calculate_dim()
        
        self.weight = Parameter.empty(self.num_features, dtype=dtype, device=device)
        self.bias = Parameter.empty(self.num_features, dtype=dtype, device=device)
        if self.track_running_stats:
            self.running_mean = Buffer.empty(self.num_features, dtype=dtype, device=device)
            self.running_var = Buffer.empty(self.num_features, dtype=dtype, device=device)
        else:
            self.running_mean = None
            self.running_var = None
        self.reset_parameters()

    def forward(self, x):
        fn = self._forward_train if self.training else self._forward_eval
        x = fn(x)
        return x
    
    def _forward_train(self, x):
        # always expecting features at the last dim.
        n = x.numel() // x.shape[-1]
        mean = x.mean(self.dim, keepdim=True)
        var_sum = ((x - mean) ** 2).sum(self.dim, keepdim=True)
        var_biased = var_sum / n
        x_norm = (x - mean) / ((var_biased + self.eps).sqrt())
        x = self.weight * x_norm + self.bias
        if self.track_running_stats:
            with no_grad():
                self.running_mean = (mean.reshape(-1) * self.momentum) + (self.running_mean * (1.0 - self.momentum))
                var_unbiased = var_sum / (n - 1)
                self.running_var = (var_unbiased.reshape(-1) * self.momentum) + (self.running_var * (1.0 - self.momentum))
        return x
    
    def _forward_eval(self, x):
        if self.running_mean is None:
            mean = x.mean(self.dim, keepdim=True)
        else:
            mean = self.running_mean
        if self.running_var is None:
            var_ = ((x - mean) ** 2).mean(self.dim, keepdim=True)
        else:
            var_ = self.running_var
        x_norm = (x - mean) / ((var_ + self.eps).sqrt())
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
        if self.track_running_stats:
            init.zeros(self.running_mean)
            init.ones(self.running_var)

    @abstractmethod
    def _calculate_dim(self): pass


class BatchNorm1d(_BatchNorm):
    """
    check 2d only
    check batch size > 1
    accumulate running

    weight: [num_features,]
    bias: [num_features,]
    """
    
    def _calculate_dim(self):
        self.dim = 0


class BatchNorm2d(_BatchNorm):
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


class LayerNorm(Module):
    
    def __init__(self, normalized_shape, eps=1e-05, dtype=None, device=None):
        super().__init__()
        normalized_shape = (normalized_shape,) if not isinstance(normalized_shape, (tuple, list)) else normalized_shape
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.dim = tuple(range(-len(self.normalized_shape), 0))
        self.weight = Parameter.empty(self.normalized_shape, dtype=dtype, device=device)
        self.bias = Parameter.empty(self.normalized_shape, dtype=dtype, device=device)
        self.reset_parameters()

    def forward(self, x):
        mean = x.mean(self.dim, keepdim=True)
        var_ = ((x - mean) ** 2).mean(self.dim, keepdim=True)
        x_norm = (x - mean) / ((var_ + self.eps).sqrt())
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
