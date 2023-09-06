from uuid import uuid4

from .const import OP, DTYPE
from .storage import StorageDispatch
from .ops import OpDispatch


class Tensor:

    @classmethod
    def empty(cls, *shape, dtype=None, device=None, requires_grad=True):
        tensor = cls._factory('empty', shape=shape, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor
    
    @classmethod
    def zeros(cls, *shape, dtype=None, device=None, requires_grad=True):
        tensor = cls._factory('zeros', shape=shape, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor
    
    @classmethod
    def ones(cls, *shape, dtype=None, device=None, requires_grad=True):
        tensor = cls._factory('ones', shape=shape, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor
    
    @classmethod
    def rand(cls, *shape, dtype=None, device=None, requires_grad=True):
        kwargs = {'a': 0.0, 'b': 1.0}
        tensor = cls._factory('random_uniform', shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, **kwargs)
        return tensor
    
    @classmethod
    def bernoulli(cls, p, shape, dtype=None, device=None, requires_grad=True):
        kwargs = {'p': p}
        tensor = cls._factory('bernoulli', shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, **kwargs)
        return tensor

    @classmethod
    def _factory(cls, method, shape, dtype, device, requires_grad, **kwargs):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        storage = StorageDispatch.get(device)
        data = getattr(storage, method)(shape=shape, dtype=dtype, **kwargs)
        tensor = cls(data=data, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor

    def __init__(
        self,
        data,
        dtype=None,
        device=None,
        name=None,
        requires_grad=True,
    ):
        self.name = name or f'tensor@{str(uuid4())[:8]}'
        self.requires_grad = requires_grad
        
        self._storage = StorageDispatch.get(device)
        self.data = self._storage.tensor(data, dtype=dtype)
        self.grad = self._storage.zeros(self.data.shape, dtype=dtype)
        
        self._children = ()
        self._op = None

    @property
    def dtype(self):
        return self._storage.get_dtype(self.data)

    @property
    def device(self):
        return self._storage.get_device(self.data)

    @property
    def shape(self):
        return self._storage.get_shape(self.data)
    
    @property
    def ndim(self):
        return len(self.shape)
    
    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def _backward(self):
        return self._op.backward if self._op is not None else lambda: None
    
    def __repr__(self):
        return f'Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device}, name={self.name})'

    def __getitem__(self, slice_):
        out = OpDispatch.execute(OP.SELECT, self, slice_=slice_)
        return out

    def __add__(self, other):
        other = self._wrap_constant_maybe(other)
        out = OpDispatch.execute(OP.ADD, self, other)
        return out

    __radd__ = __add__

    def __sub__(self, other):
        other = self._wrap_constant_maybe(other)
        out = OpDispatch.execute(OP.SUB, self, other)
        return out

    def __rsub__(self, other):
        other = self._wrap_constant_maybe(other)
        return other - self
    
    def __mul__(self, other):
        other = self._wrap_constant_maybe(other)
        out = OpDispatch.execute(OP.MUL, self, other)
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._wrap_constant_maybe(other)
        out = OpDispatch.execute(OP.DIV, self, other)
        return out

    def __rtruediv__(self, other):
        other = self._wrap_constant_maybe(other)
        return other / self

    def __pow__(self, value):
        assert isinstance(value, (int, float))
        out = OpDispatch.execute(OP.POW, self, value=value)
        return out
    
    def __neg__(self):
        neg = self._wrap_constant_maybe(-1.0)
        out = OpDispatch.execute(OP.MUL, self, neg)
        return out

    def __invert__(self):
        out = OpDispatch.execute(OP.INVERT, self)
        return out

    def sqrt(self):
        out = self ** 0.5
        return out

    def exp(self):
        out = OpDispatch.execute(OP.EXP, self)
        return out

    def log(self):
        out = OpDispatch.execute(OP.LOG, self)
        return out

    def sum(self, dim=None, keepdim=False):
        out = OpDispatch.execute(OP.SUM_REDUCE, self, dim=dim, keepdim=keepdim)
        return out
    
    def mean(self, dim=None, keepdim=False):
        out = OpDispatch.execute(OP.MEAN_REDUCE, self, dim=dim, keepdim=keepdim)
        return out
    
    def max(self, dim=None):
        out = OpDispatch.execute(OP.MAX_REDUCE, self, dim=dim)
        return out

    def min(self, dim=None):
        out = OpDispatch.execute(OP.MIN_REDUCE, self, dim=dim)
        return out

    def relu(self):
        out = OpDispatch.execute(OP.RELU, self)
        return out
    
    def sigmoid(self):
        out = OpDispatch.execute(OP.SIGMOID, self)
        return out
    
    def softmax(self, dim):
        out = OpDispatch.execute(OP.SOFTMAX, self, dim=dim)
        return out

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        out = OpDispatch.execute(OP.RESHAPE, self, shape=shape)
        return out
    
    def permute(self, *dims):
        dims = dims[0] if len(dims) == 1 and isinstance(dims[0], (tuple, list)) else dims
        out = OpDispatch.execute(OP.PERMUTE, self, dims=dims)
        return out
    
    def transpose(self, *dims):
        dims = dims[0] if len(dims) == 1 and isinstance(dims[0], (tuple, list)) else dims
        out = OpDispatch.execute(OP.PERMUTE, self, dims=dims)
        return out

    def squeeze(self, dim):
        out = OpDispatch.execute(OP.SQUEEZE, self, dim=dim)
        return out

    def unsqueeze(self, dim):
        out = OpDispatch.execute(OP.UNSQUEEZE, self, dim=dim)
        return out
    
    def concat(self, tensors, dim):
        tensors = [self, *tensors]
        out = OpDispatch.execute(OP.CONCAT, tensors, dim=dim)
        return out
    
    def cat(self, tensors, dim):
        out = self.concat(tensors, dim=dim)
        return out

    def masked_fill_(self, mask, value):
        out = OpDispatch.execute(OP.MASKED_FILL, self, mask=mask, value=value)
        return out
    
    def lookup(self, mask):
        out = OpDispatch.execute(OP.LOOKUP, self, mask=mask)
        return out

    def matmul(self, other):
        out = OpDispatch.execute(OP.MATMUL, self, other)
        return out
    
    def conv2d(self, kernel, bias=None, stride=None, padding=None):
        children = (self, kernel, bias) if bias is not None else (self, kernel)
        out = OpDispatch.execute(OP.CONV2D, *children, stride=stride, padding=padding)
        return out
    
    def max_pool2d(self, kernel_size, stride=None, padding=None):
        children = (self,)
        out = OpDispatch.execute(OP.MAX_POOL2D, *children, kernel_size=kernel_size, stride=stride, padding=padding)
        return out
    
    def avg_pool2d(self, kernel_size, stride=None, padding=None):
        children = (self,)
        out = OpDispatch.execute(OP.AVG_POOL2D, *children, kernel_size=kernel_size, stride=stride, padding=padding)
        return out

    def backward(self, upstream=None):
        if upstream is not None:
            upstream = self._storage.tensor(upstream, dtype=self.dtype)
        else:
            upstream = self._storage.ones(self.shape, dtype=self.dtype)
        self.grad = upstream
        nodes = self._traverse()
        for node in reversed(nodes):
            node._backward()

    def copy(self):
        ob = self._copy_from_data(self.data)
        return ob
    
    def detach(self):
        ob = self.copy()
        return ob

    def from_data(self, data):
        ob = self._copy_from_data(data)
        return ob

    def zeros_like(self):
        data = self._storage.zeros(self.data.shape, dtype=self.dtype)
        return self._copy_from_data(data)
    
    def ones_like(self):
        data = self._storage.ones(self.data.shape, dtype=self.dtype)
        return self._copy_from_data(data)
    
    def float(self):
        data = self._storage.cast(self.data, dtype=DTYPE.FLOAT32)
        return self._copy_partial(data=data, dtype=DTYPE.FLOAT32)
    
    def bool(self):
        data = self._storage.cast(self.data, dtype=DTYPE.BOOL)
        return self._copy_partial(data=data, dtype=DTYPE.BOOL)

    def arange(self, n):
        data = self._storage.arange(n, dtype=self.dtype)
        return self._copy_from_data(data)

    def tolist(self):
        return self.data.tolist()
    
    def item(self):
        return self.tolist()

    def render(self):
        from .render import render_graph
        render_graph(self)

    def _copy_partial(self, data=None, dtype=None, device=None, requires_grad=None):
        ob = type(self)(
            data=data if data is not None else self.data.copy(),
            dtype=dtype or self.dtype,
            device=device or self.device,
            requires_grad=requires_grad or self.requires_grad,
        )
        return ob

    def _copy_from_data(self, data):
        ob = type(self)(
            data=data,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        return ob

    def _wrap_constant_maybe(self, value):
        if isinstance(value, (float, int)):
            value = self._storage.tensor(value, dtype=self.dtype)
            tensor = self._copy_from_data(value)
            tensor.requires_grad = False
            value = tensor
        return value
    
    def _traverse(self):
        nodes_sorted = []
        visited = set()
        
        def __traverse(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for ch in node._children:
                __traverse(ch)
            nodes_sorted.append(node)
        
        __traverse(self)
        return nodes_sorted
