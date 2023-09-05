import operator
import functools

from .util import get_numpy
from ..stubs import ReduceOp
from ..dispatch import OpDispatch
from ...const import OP, DEVICE


class _KeepdimMixin:

    def _get_reduced_dims(self):
        if self.dim is None:
            dims = list(range(self.x.ndim))
        elif isinstance(self.dim, int):
            dims = (self.dim,)
        else:
            dims = self.dim
        return dims
    
    def _get_reduced_size(self):
        dims = self._get_reduced_dims()
        sizes = [self.x.shape[d] for d in dims]
        size = functools.reduce(operator.mul, sizes, 1)
        return size
    
    def _apply_keepdim_to_upstream_grad(self):
        dims = self._get_reduced_dims()
        u = self.out.grad
        if not self.keepdim:
            shape = list(self.x.shape)
            for d in dims:
                shape[d] = 1
            u = self.out.grad.reshape(shape)
        return u


@OpDispatch.register(OP.SUM_REDUCE, DEVICE.CPU)
class SumReduce(ReduceOp, _KeepdimMixin):
    
    def forward(self):
        data = self.x.data.sum(self.dim, keepdims=self.keepdim)
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            u = self._apply_keepdim_to_upstream_grad()
            self.x.grad += u


@OpDispatch.register(OP.MEAN_REDUCE, DEVICE.CPU)
class MeanReduce(ReduceOp, _KeepdimMixin):

    def forward(self):
        data = self.x.data.mean(self.dim, keepdims=self.keepdim)
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            size = self._get_reduced_size()
            u = self._apply_keepdim_to_upstream_grad()
            self.x.grad += 1.0 / size * u
    

class _MinMaxReduce(ReduceOp):
    _FUNC = None

    def __init__(self, x, *, dim=None):
        super().__init__(x, dim=dim)
        if isinstance(self.dim, (tuple, list)):
            msg = f'multi-dimensional min-max reduce is not supported.'
            raise Exception(msg)
        self.mask = None
        self.np = get_numpy()
        self._func = self._FUNC
        self._argfunc = f'arg{self._func}'

    def forward(self):
        x = self.x.data
        argfunc = getattr(x, self._argfunc)
        self.mask = argfunc(self.dim, keepdims=True)
        func = getattr(x, self._func)
        data = func(self.dim)
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            u = self.out.grad
            g = self.np.zeros_like(self.x.data)
            if self.dim is None:
                ix = self.mask.item()
                ix = self.np.unravel_index(ix, self.x.shape)
                g[ix] = 1.0
            else:
                self.np.put_along_axis(g, self.mask, 1.0, self.dim)
                u = self.np.expand_dims(u, self.dim)
            g *= u
            self.x.grad += g


@OpDispatch.register(OP.MAX_REDUCE, DEVICE.CPU)
class MaxReduce(_MinMaxReduce):
    _FUNC = 'max'


@OpDispatch.register(OP.MIN_REDUCE, DEVICE.CPU)
class MinReduce(_MinMaxReduce):
    _FUNC = 'min'
