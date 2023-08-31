from .util import get_numpy
from ..stubs import ReduceOp
from ..dispatch import OpDispatch
from ...const import OP, DEVICE


@OpDispatch.register(OP.SUM_REDUCE, DEVICE.CPU)
class SumReduce(ReduceOp):
    
    def forward(self):
        data = self.x.data.sum(self.dim)
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            np = get_numpy()
            out_grad = self.out.grad
            if self.dim is None:
                out_grad = self.out.grad
            elif self.dim == 0 and self.x.ndim < 2:
                out_grad = self.out.grad
            else:
                out_grad = np.expand_dims(self.out.grad, self.dim)
            self.x.grad += out_grad


@OpDispatch.register(OP.MEAN_REDUCE, DEVICE.CPU)
class MeanReduce(ReduceOp):

    def forward(self):
        data = self.x.data.mean(self.dim)
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            np = get_numpy()
            size = self.x.numel() if self.dim is None else self.x.shape[self.dim]
            if self.dim is None:
                out_grad = self.out.grad
            elif self.dim == 0 and self.x.ndim < 2:
                out_grad = self.out.grad
            else:
                out_grad = np.expand_dims(self.out.grad, self.dim)
            self.x.grad += 1.0 / size * out_grad
    

class _MinMaxReduce(ReduceOp):
    _FUNC = None

    def __init__(self, x, *, dim=None):
        super().__init__(x, dim=dim)
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


@OpDispatch.register(OP.STD_REDUCE, DEVICE.CPU)
class StdReduce(ReduceOp):

    def __init__(self, x, *, dim=None):
        super().__init__(x, dim=dim)
        self.np = get_numpy()
        self._head = None
        self._tail = None

    def forward(self):
        x = self.x.detach()
        mean = x.mean(self.dim)
        mean = mean.unsqueeze(self.dim) if self.dim is not None else mean
        var_ = (x - mean) ** 2
        n = x.shape[self.dim] if self.dim is not None else x.numel()
        n -= 1
        std = (var_.sum(self.dim) / n).sqrt()
        self._head = x
        self._tail = std
        self.out = std.detach()
        return self.out

    def backward(self):
        if self.x.requires_grad:
            self._tail.backward(self.out.grad)
            self.x.grad += self._head.grad
