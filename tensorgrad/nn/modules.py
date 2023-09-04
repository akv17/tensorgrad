from .base import Module, Parameter
from . import init


class Linear(Module):
    """
    weight: [out, in]
    bias: [out,]
    """
    
    def __init__(self, in_features, out_features, bias=True, dtype=None, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = Parameter.empty(shape=(self.out_features, self.in_features), dtype=dtype, device=device)
        self.bias = Parameter.empty(shape=(self.out_features,), dtype=dtype, device=device) if bias else None
        self.reset_parameters()

    def forward(self, x):
        w = self.weight.transpose(1, 0)
        b = self.bias if self.bias is not None else None
        out = x.matmul(w) + b if b is not None else x.matmul(w)
        return out

    def init_from_torch(self, module):
        self.weight = Parameter(
            module.weight.detach().cpu().numpy(),
            dtype=self.weight.dtype,
            device=self.weight.device
        )
        if module.bias is not None:
            self.bias = Parameter(
                module.bias.detach().cpu().numpy(),
                dtype=self.bias.dtype,
                device=self.bias.device
            )

    def reset_parameters(self):
        init.uniform_fan_in(self.weight)
        if self.bias is not None:
            init.uniform_fan_in(self.bias)


class Conv2d(Module):
    """
    maybe check params against each other.

    weight: [co, ci, kh, kw]
    bias: [co,]
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        self._normalize_kernel_size()
        self._normalize_stride()
        self._normalize_padding()

        self.weight = Parameter.empty((self.out_channels, self.in_channels, *self._kernel_size), dtype=dtype, device=device)
        self.bias = Parameter.empty((self.out_channels,), dtype=dtype, device=device) if bias else None

    def forward(self, x):
        out = x.conv2d(kernel=self.weight, bias=self.bias, stride=self._stride, padding=self._padding)
        return out

    def init_from_torch(self, module):
        self.weight = Parameter(
            module.weight.detach().cpu().numpy(),
            dtype=self.weight.dtype,
            device=self.weight.device
        )
        if module.bias is not None:
            self.bias = Parameter(
                module.bias.detach().cpu().numpy(),
                dtype=self.bias.dtype,
                device=self.bias.device
            )
    
    def reset_parameters(self):
        init.uniform_fan_in(self.weight)
        if self.bias is not None:
            init.uniform_fan_in(self.bias)

    def _normalize_kernel_size(self):
        ks = self.kernel_size
        self._kernel_size = (ks, ks) if isinstance(ks, int) else ks

    def _normalize_stride(self):
        s = self.stride
        self._stride = (s, s) if isinstance(s, int) else s
    
    def _normalize_padding(self):
        p = self.padding
        if isinstance(p, int):
            self._padding = (p, p)
        elif p == 'same':
            kh, kw = self._kernel_size
            ph = (kh - 1) // 2
            pw = (kw - 1) // 2
            self._padding = (ph, pw)
        elif p == 'valid':
            self._padding = (0, 0)
        else:
            self._padding = p


class _GeneralizedPool2d(Module):
    """
    maybe check params against each other.
    """

    _OP = None
    
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._op = f'{self._OP.lower()}_pool2d'

        self._normalize_kernel_size()
        self._normalize_stride()
        self._normalize_padding()

    def forward(self, x):
        op = getattr(x, self._op)
        out = op(kernel_size=self._kernel_size, stride=self._stride, padding=self._padding)
        return out

    def init_from_torch(self, module):
        pass
    
    def reset_parameters(self):
        pass

    def _normalize_kernel_size(self):
        value = self.kernel_size
        self._kernel_size = (value, value) if isinstance(value, int) else value
    
    def _normalize_stride(self):
        value = self.stride
        if value is None:
            self._stride = self._kernel_size
        else:
            self._stride = (value, value) if isinstance(value, int) else value
    
    def _normalize_padding(self):
        value = self.padding
        self._padding = (value, value) if isinstance(value, int) else value


class MaxPool2d(_GeneralizedPool2d):
    _OP = 'max'


class AvgPool2d(_GeneralizedPool2d):
    _OP = 'avg'


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
        mean = x.mean(self.dim).unsqueeze(self.dim)
        std = ((x - mean) ** 2).mean(self.dim).unsqueeze(self.dim)
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

        self.dim = 0
        self.weight = Parameter.empty((self.num_features,), dtype=dtype, device=device)
        self.bias = Parameter.empty((self.num_features,), dtype=dtype, device=device)

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


class ReLU(Module):
    
    def forward(self, x):
        x = x.relu()
        return x
    
    def initialize(self):
        pass
    
    def parameters(self):
        return []


class Identity(Module):

    def forward(self, x):
        return x
    
    def initialize(self):
        pass
    
    def parameters(self):
        return []


class Sigmoid(Module):

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
    