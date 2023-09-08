import operator
import functools

from .util.np import NumpyProvider
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
    

class _MinMaxReduce(ReduceOp, NumpyProvider):
    # this implementation is pretty obscure because of interoperability between numpy and cupy.
    # in numpy this could have been done easily via argmax and put_along_axis in backward.
    # in cupy there is no put_along_axis so we need another implementation.
    # forward:
    # 1. move target dim as the last dim
    # 2. reshape to 2d
    # 3. compute argfunc over last dim
    # backward:
    # 1. transpose local grad
    # 2. reshape local grad to 2d
    # 3. fill ones into argfunc mask
    # 4. broadcast multiply with upstream `g[i, :] *= u[i]`
    # 5. reshape and transpose to original shape

    _FUNC = None

    def __init__(self, x, *, dim=None):
        super().__init__(x, dim=dim)
        if isinstance(self.dim, (tuple, list)):
            msg = f'multi-dimensional min-max reduce is not supported.'
            raise Exception(msg)
        
        if self.dim is not None:
            self.dim = self.x.ndim - abs(self.dim) if self.dim < 0 else self.dim
            self._backward_compute_dims_and_shapes()
        
        self.mask = None
        self._func = self._FUNC
        self._argfunc = f'arg{self._func}'

    def forward(self):
        x = self.x.data
        if self.dim is not None:
            _x = self.np.transpose(x, self._dimt)
            _x = _x.reshape(-1, _x.shape[-1])
            argfunc = getattr(_x, self._argfunc)
            self.mask = argfunc(-1)
        else:
            argfunc = getattr(x, self._argfunc)
            self.mask = argfunc()
        func = getattr(x, self._func)
        o = func(self.dim)
        self.out = self.x.from_data(o)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            if self.dim is None:
                m = self.mask
                u = self.out.grad
                g = self.np.zeros_like(self.x.data)
                i = self.np.unravel_index(m, g.shape)
                g[i] = u
            else:
                m = self.mask
                u = self.out.grad
                u = u.ravel()
                u = self.np.expand_dims(u, -1)
                
                g = self.np.zeros_like(self.x.data)
                g = self.np.transpose(g, self._dimt)
                g = g.reshape(-1, g.shape[-1])
                
                g[range(len(g)), m] = 1.0
                g *= u
                g = g.reshape(self._shapet)
                g = self.np.transpose(g, self._dimo)
            self.x.grad += g
    
    def _backward_compute_dims_and_shapes(self):
        dim = self.dim
        dim_last = self.x.ndim - 1
        if dim == dim_last:
            dimt = list(range(self.x.ndim))
            dimo = dimt
            shapet = self.x.shape
        else:
            dimt = list(range(self.x.ndim))
            dimt.pop(dim)
            dimt.append(dim)
            shapet = [self.x.shape[d] for d in dimt]
            dimo = [dimt.index(d) for d in range(self.x.ndim)]
        self._dimt = dimt
        self._dimo = dimo
        self._shapet = shapet


@OpDispatch.register(OP.MAX_REDUCE, DEVICE.CPU)
class MaxReduce(_MinMaxReduce):
    _FUNC = 'max'


@OpDispatch.register(OP.MIN_REDUCE, DEVICE.CPU)
class MinReduce(_MinMaxReduce):
    _FUNC = 'min'
