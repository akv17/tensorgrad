import math
from .base import Module, Parameter
from .. import init
from ...const import DTYPE


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
        return self

    def reset_parameters(self):
        init.uniform_fan_in(self.weight)
        if self.bias is not None:
            init.uniform_fan_in(self.bias)

    def _check_input(self, x):
        n_in = x.shape[-1]
        if n_in != self.in_features:
            msg = f'expected input to have {self.in_features} features but got {n_in} features.'
            raise Exception(msg)


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
        self.reset_parameters()

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
        return self
    
    def reset_parameters(self):
        init.uniform_fan_in(self.weight)
        if self.bias is not None:
            init.uniform_fan_in(self.bias)

    def _check_input(self, x):
        n_in = x.shape[1]
        if n_in != self.in_channels:
            msg = f'expected input to have {self.in_channels} channels but got {n_in} channels.'
            raise Exception(msg)

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
        self.reset_parameters()
    
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
        return self

    def _check_input(self, query, key, value, attn_mask=None):
        if query.ndim != 3:
            msg = f'expected query to have 3 dimensions but got {query.ndim}.'
            raise Exception(msg)
        if key.ndim != 3:
            msg = f'expected key to have 3 dimensions but got {key.ndim}.'
            raise Exception(msg)
        if value.ndim != 3:
            msg = f'expected value to have 3 dimensions but got {value.ndim}'
            raise Exception(msg)
        if attn_mask is not None and attn_mask.dtype is not DTYPE.BOOL:
            msg = f'expected attn_mask of type {DTYPE.BOOL} but got {attn_mask.dtype}.'
            raise Exception(msg)
        if attn_mask is not None and attn_mask.ndim not in (2, 3):
            msg = f'expected attn_mask to have 2 or 3 dimensions but got {attn_mask.ndim}.'
            raise Exception(msg)
        if key.shape[1] != value.shape[1]:
            msg = f'expected key and value to have same sequence length but got {(key.shape[1], value.shape[1])}.'
            raise Exception(msg)
        if key.shape[-1] != value.shape[-1]:
            msg = f'expected key and value to have same embedding dimensions but got {(key.shape[-1], value.shape[-1])}.'
            raise Exception(msg)
