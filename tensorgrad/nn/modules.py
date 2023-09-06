import math
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

        self.weight = Parameter.empty((self.out_features, self.in_features), dtype=dtype, device=device)
        self.bias = Parameter.empty((self.out_features,), dtype=dtype, device=device) if bias else None
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


class MultiheadAttention(Module):
    """
    if attn_mask 3d assumed shape is (bs, sl, sl) so the same mask is broadcasted over all the heads.
    """

    def __init__(self, embed_dim, num_heads, dtype=None, device=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if self.embed_dim % self.num_heads != 0:
            msg = '`embed_dim` must be divisible by `num_heads`'
            raise Exception(msg)
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_weight = Parameter.empty((self.embed_dim, self.embed_dim), dtype=dtype, device=device)
        self.k_weight = Parameter.empty((self.embed_dim, self.embed_dim), dtype=dtype, device=device)
        self.v_weight = Parameter.empty((self.embed_dim, self.embed_dim), dtype=dtype, device=device)
        self.o_weight = Parameter.empty((self.embed_dim, self.embed_dim), dtype=dtype, device=device)
    
    def forward(self, query, key, value, attn_mask=None):
        q = query.matmul(self.q_weight.transpose(1, 0))
        k = key.matmul(self.k_weight.transpose(1, 0))
        v = value.matmul(self.v_weight.transpose(1, 0))

        bs, q_sl = q.shape[:2]
        bs, kv_sl = k.shape[:2]
        q = q.reshape(bs, q_sl, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.reshape(bs, kv_sl, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.reshape(bs, kv_sl, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)

        qk = q.matmul(k.permute(0, 1, 3, 2))
        qk /= math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            attn_mask_bool = attn_mask
            attn_mask = attn_mask.float()
            attn_mask.masked_fill_(attn_mask_bool, -math.inf)
            if attn_mask.ndim == 2:
                # broadcast over `batch_size` and `num_heads`.
                attn_mask = attn_mask.reshape(1, 1, *attn_mask.shape)
            elif attn_mask.ndim == 3:
                # broadcast over `num_heads`.
                attn_mask = attn_mask.unsqueeze(1)
            qk += attn_mask
        qk = qk.softmax(-1)

        attn = qk.matmul(v)
        attn = attn.permute(0, 2, 1, 3)
        attn = attn.reshape(bs, q_sl, self.num_heads * self.head_dim)

        out = attn.matmul(self.o_weight.transpose(1, 0))
        return out
    
    def reset_parameters(self):
        init.uniform_fan_in(self.q_weight)
        init.uniform_fan_in(self.k_weight)
        init.uniform_fan_in(self.v_weight)
        init.uniform_fan_in(self.o_weight)
    
    def init_from_torch(self, module):
        params = dict(module.named_parameters())
        out_weight = params['out_proj.weight'].detach().cpu().numpy()
        in_weights = params['in_proj_weight'].detach().cpu().numpy()
        # torch q, k, v weights are fused along first dim so reshape by 3 to unfuse them.
        q_weight, k_weight, v_weight = in_weights.reshape(3, self.embed_dim, self.embed_dim)
        dtype = self.q_weight.dtype
        device = self.q_weight.device
        self.q_weight = Parameter(q_weight, dtype=dtype, device=device)
        self.k_weight = Parameter(k_weight, dtype=dtype, device=device)
        self.v_weight = Parameter(v_weight, dtype=dtype, device=device)
        self.o_weight = Parameter(out_weight, dtype=dtype, device=device)


class Embedding(Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, dtype=None, device=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = Parameter.empty((self.num_embeddings, self.embedding_dim), dtype=dtype, device=device)
    
    def forward(self, x):
        o = self.weight.lookup(x)
        return o
    
    def reset_parameters(self):
        init.uniform_fan_in(self.weight)
    
    def init_from_torch(self, module):
        self.weight = Parameter(
            module.weight.detach().cpu().numpy(),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )


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
    