from uuid import uuid4

from .const import DTYPE, OP, BACKEND
from .backend import BackendDispatch
from .ops import OpDispatch


class Tensor:

    def __init__(
        self,
        data,
        dtype=DTYPE.FLOAT32,
        backend=BACKEND.NUMPY,
        name=None,
        requires_grad=True,
    ):
        self.backend = backend
        self._backend = BackendDispatch.get(self.backend)
        
        self.dtype = dtype
        self.name = name or f'tensor@{str(uuid4())[:8]}'
        self.requires_grad = requires_grad
        
        self.data = self._backend.tensor(data, dtype=self.dtype)
        self.grad = self._backend.zeros(self.data.shape, dtype=self.dtype)
        
        self._children = ()
        self._op = None

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return len(self.shape)

    @property
    def _backward(self):
        return self._op.backward if self._op is not None else lambda: None
    
    def __repr__(self):
        return f'Tensor(name={self.name}, shape={self.shape}, dtype={self.dtype}, backend={self.backend})'

    def __add__(self, other):
        out = OpDispatch.execute(OP.ADD, self, other)
        return out

    __radd__ = __add__

    def __sub__(self, other):
        out = OpDispatch.execute(OP.SUB, self, other)
        return out

    def __rsub__(self, other):
        return other - self
    
    def __mul__(self, other):
        out = OpDispatch.execute(OP.MUL, self, other)
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        out = OpDispatch.execute(OP.DIV, self, other)
        return out

    def __rtruediv__(self, other):
        return other / self

    def __pow__(self, value):
        assert isinstance(value, (int, float))
        out = OpDispatch.execute(OP.POW, self, value=value)
        return out

    def exp(self):
        out = OpDispatch.execute(OP.EXP, self)
        return out

    def log(self):
        out = OpDispatch.execute(OP.LOG, self)
        return out

    def sum(self, dim=None):
        out = OpDispatch.execute(OP.SUM_REDUCE, self, dim=dim)
        return out
    
    def mean(self, dim=None):
        out = OpDispatch.execute(OP.MEAN_REDUCE, self, dim=dim)
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

    def reshape(self, shape):
        out = OpDispatch.execute(OP.RESHAPE, self, shape=shape)
        return out
    
    def permute(self, dims):
        out = OpDispatch.execute(OP.PERMUTE, self, dims=dims)
        return out

    def squeeze(self, dim):
        out = OpDispatch.execute(OP.SQUEEZE, self, dim=dim)
        return out

    def unsqueeze(self, dim):
        out = OpDispatch.execute(OP.UNSQUEEZE, self, dim=dim)
        return out
    
    def matmul(self, other):
        out = OpDispatch.execute(OP.MATMUL, self, other)
        return out

    def backward(self, upstream=None):
        if upstream is not None:
            upstream = self._backend.tensor(upstream, dtype=self.dtype)
        else:
            upstream = self._backend.ones(self.shape, dtype=self.dtype)
        self.grad = upstream
        visited = set()
        nodes_sorted = []

        def _traverse(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for ch in node._children:
                _traverse(ch)
            nodes_sorted.append(node)
        
        _traverse(self)
        for node in reversed(nodes_sorted):
            node._backward()

    def copy(self):
        ob = self._copy_from_data(self.data)
        return ob

    def from_data(self, data):
        ob = self._copy_from_data(data)
        return ob

    def zeros_like(self):
        data = self._backend.zeros(self.data.shape, dtype=self.dtype)
        return self._copy_from_data(data)
    
    def zeros(self, shape):
        data = self._backend.zeros(shape, dtype=self.dtype)
        return self._copy_from_data(data)
    
    def ones_like(self):
        data = self._backend.ones(self.data.shape, dtype=self.dtype)
        return self._copy_from_data(data)
    
    def tolist(self):
        return self.data.tolist()

    def render(self):
        from .render import render_graph
        render_graph(self)

    def _copy_from_data(self, data, name=None):
        ob = type(self)(
            data=data,
            dtype=self.dtype,
            name=name,
            requires_grad=self.requires_grad,
            backend=self.backend
        )
        return ob
