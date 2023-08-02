from uuid import uuid4

from .const import DTYPE, OP, BACKEND
from .backend import get_backend


class Tensor:
    
    def __init__(
        self,
        data,
        dtype=None,
        name=None,
        requires_grad=True,
        backend=BACKEND.NUMPY,
    ):
        self.backend = backend
        self._backend = get_backend(self.backend)
        
        self.dtype = dtype or DTYPE.FLOAT32
        self.name = name or f'tensor@{str(uuid4())[:8]}'
        self.requires_grad = requires_grad
        
        self.data = self._backend.tensor(data, dtype=self.dtype)
        self.grad = self._backend.zeros(self.data.shape, dtype=self.dtype)
        
        self._backward = lambda: None
        self._children = ()
        self._op = None

    @property
    def shape(self):
        return self.data.shape
    
    def __repr__(self):
        return f'Tensor(name={self.name}, shape={self.shape}, dtype={self.dtype}, backend={self.backend})'

    def __add__(self, other):
        out_data = self.data + other.data
        out = self._copy_from_data(out_data)
        out._children = (self, other)
        out._op = OP.ADD

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        
        out._backward = _backward
        return out

    __radd__ = __add__

    def __sub__(self, other):
        out_data = self.data - other.data
        out = self._copy_from_data(out_data)
        out._children = (self, other)
        out._op = OP.SUB

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += -out.grad
        
        out._backward = _backward
        return out

    def __rsub__(self, other):
        return other - self
    
    def __mul__(self, other):
        out_data = self.data * other.data
        out = self._copy_from_data(out_data)
        out._children = (self, other)
        out._op = OP.MUL

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        out_data = self.data / other.data
        out = self._copy_from_data(out_data)
        out._children = (self, other)
        out._op = OP.DIV

        def _backward():
            if self.requires_grad:
                self.grad += 1.0 / other.data * out.grad
            if other.requires_grad:
                other.grad += (-self.data / (other.data ** 2)) * out.grad
        
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return other / self

    def __pow__(self, value):
        assert isinstance(value, (int, float))
        out_data = self.data ** value
        out = self._copy_from_data(out_data)
        out._children = (self,)
        out._op = OP.POW

        def _backward():
            if self.requires_grad:
                n = value
                self.grad += (n * self.data ** (n-1)) * out.grad
        
        out._backward = _backward
        return out

    def exp(self):
        out_data = self.data.exp()
        out = self._copy_from_data(out_data)
        out._children = (self,)
        out._op = OP.EXP

        def _backward():
            if self.requires_grad:
                self.grad += out_data * out.grad
        
        out._backward = _backward
        return out

    def log(self):
        out_data = self.data.log()
        out = self._copy_from_data(out_data)
        out._children = (self,)
        out._op = OP.LOG

        def _backward():
            if self.requires_grad:
                self.grad += 1 / self.data * out.grad
        
        out._backward = _backward
        return out

    def sum(self, dim=0):
        out_data = self.data.sum(dim=dim)
        out = self._copy_from_data(out_data)
        out._children = (self,)
        out._op = OP.SUM_REDUCE

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
        
        out._backward = _backward
        return out

    def backward(self):
        self.grad = self._backend.ones(self.shape, dtype=self.dtype)
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
