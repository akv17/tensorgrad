import math

import numba

from .util import get_numpy, get_numba
from ..stubs import BaseOp
from ..dispatch import OpDispatch
from ...const import OP, DEVICE


@OpDispatch.register(OP.CONV2D, DEVICE.CPU)
class Conv2D(BaseOp):

    def __init__(self, x, kernel, bias=None, *, stride=None, padding=None):
        self.out = None
        self.x = x
        self.kernel = kernel
        self.bias = bias
        self.stride = stride
        self.padding = padding
        if self.stride == (1, 1):
            self.stride = None
        if self.padding == (0, 0):
            self.padding = None

    def forward(self):
        bias = self.bias.data if self.bias is not None else None
        data = _nb_conv2d_forward(
            x=self.x.data,
            kernel=self.kernel.data,
            bias=bias,
            stride=self.stride,
            padding=self.padding
        )
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            g = _nb_conv2d_backward_x(
                x=self.x.data,
                kernel=self.kernel.data,
                upstream=self.out.grad,
                stride=self.stride,
                padding=self.padding
            )
            self.x.grad += g
        
        if self.kernel.requires_grad:
            g = _nb_conv2d_backward_k(
                x=self.x.data,
                kernel=self.kernel.data,
                upstream=self.out.grad,
                stride=self.stride,
                padding=self.padding
            )
            self.kernel.grad += g
        
        if self.bias is not None and self.bias.requires_grad:
            g = _np_conv2d_backward_b(upstream=self.out.grad)
            self.bias.grad += g


def _conv2d_compute_output_size(x, kernel, stride):
    ih, iw = x.shape[-2:]
    kh, kw = kernel.shape[-2:]
    sh, sw = stride
    oh = math.floor((ih - kh) / sh + 1)
    ow = math.floor((iw - kw) / sw + 1)
    return oh, ow


def _nb_conv2d_forward(x, kernel, bias, stride, padding):
    np = get_numpy()
    
    k = kernel
    co = k.shape[0]
    b = np.zeros((co,), dtype=k.dtype) if bias is None else bias
    if padding is not None:
        ph, pw = padding
        x = np.pad(x, [(0, 0), (0, 0), (ph, ph), (pw, pw)])
    bs = x.shape[0]
    stride = stride or (1, 1)
    sh, sw = stride
    oh, ow = _conv2d_compute_output_size(x=x, kernel=k, stride=stride)
    x = np.transpose(x, [0, 2, 3, 1])
    k = np.transpose(k, [2, 3, 1, 0])
    o = np.zeros((bs, oh, ow, co))
    o = _nb_jit_conv2d_forward(x, k, b, o, sh, sw)
    o = np.transpose(o, [0, 3, 1, 2])
    return o


@numba.jit(nopython=True, parallel=True)
def _nb_jit_conv2d_forward(x, k, b, o, sh, sw):
    bs = x.shape[0]
    oh = o.shape[1]
    ow = o.shape[2]
    kh = k.shape[1]
    kw = k.shape[2]
    ci = k.shape[0]
    co = k.shape[3]

    niter = bs * oh * ow
    bss = oh * ow
    ohs = ow
    ows = 1
    for i in numba.prange(niter):
        bi = int(i / bss)
        oi = int((i - bi * bss) / ohs)
        oj = int((i - (bi * bss + oi * ohs)) / ows)
        for ki in range(kh):
            for kj in range(kw):
                ii = (oi * sh) + ki
                ij = (oj * sw) + kj
                for m in range(co):
                    for p in range(ci):
                        xp = x[bi, ii, ij, p]
                        kp = k[ki, kj, p, m]
                        zi = xp * kp
                        o[bi, oi, oj, m] += zi
                    o[bi, oi, oj, m] += b[m] / (kh * kw)
    return o


def _nb_conv2d_backward_x(x, kernel, upstream, stride, padding):
    np = get_numpy()
    k = kernel
    u = upstream
    if padding is not None:
        ph, pw = padding
        x = np.pad(x, [(0, 0), (0, 0), (ph, ph), (pw, pw)])
    g = np.zeros(x.shape, dtype=x.dtype)
    stride = stride or (1, 1)
    sh, sw = stride
    x = np.transpose(x, [0, 2, 3, 1])
    g = np.transpose(g, [0, 2, 3, 1])
    k = np.transpose(k, [2, 3, 1, 0])
    u = np.transpose(u, [0, 2, 3, 1])
    g = _nb_jit_conv2d_backward_x(x, k, u, g, sh, sw)
    if padding is not None:
        ph, pw = padding
        g = g[:, ph:-ph, pw:-pw, :]
    g = np.transpose(g, [0, 3, 1, 2])
    return g


@numba.jit(nopython=True, parallel=False)
def _nb_jit_conv2d_backward_x(x, k, u, g, sh, sw):
    bs = x.shape[0]
    oh = u.shape[1]
    ow = u.shape[2]
    kh = k.shape[1]
    kw = k.shape[2]
    ci = k.shape[0]
    co = k.shape[3]
    for bi in range(bs):
        for oi in range(oh):
            for oj in range(ow):
                for ki in range(kh):
                    for kj in range(kw):
                        ii = (oi * sh) + ki
                        ij = (oj * sw) + kj
                        for m in range(co):
                            for p in range(ci):
                                ui = u[bi, oi, oj, m]
                                zi = k[ki, kj, p, m]
                                gi = zi * ui
                                g[bi, ii, ij, p] += gi
    return g


def _nb_conv2d_backward_k(x, kernel, upstream, stride, padding):
    np = get_numpy()
    k = kernel
    u = upstream
    if padding is not None:
        ph, pw = padding
        x = np.pad(x, [(0, 0), (0, 0), (ph, ph), (pw, pw)])
    g = np.zeros(k.shape, dtype=k.dtype)
    stride = stride or (1, 1)
    sh, sw = stride
    x = np.transpose(x, [0, 2, 3, 1])
    k = np.transpose(k, [2, 3, 1, 0])
    g = np.transpose(g, [2, 3, 1, 0])
    u = np.transpose(u, [0, 2, 3, 1])
    g = _nb_jit_conv2d_backward_k(x, k, u, g, sh, sw)
    g = np.transpose(g, [3, 2, 0, 1])
    return g


@numba.jit(nopython=True, parallel=False)
def _nb_jit_conv2d_backward_k(x, k, u, g, sh, sw):
    bs = x.shape[0]
    oh = u.shape[1]
    ow = u.shape[2]
    kh = k.shape[1]
    kw = k.shape[2]
    ci = k.shape[0]
    co = k.shape[3]
    for bi in range(bs):
        for oi in range(oh):
            for oj in range(ow):
                for ki in range(kh):
                    for kj in range(kw):
                        ii = (oi * sh) + ki
                        ij = (oj * sw) + kj
                        for m in range(co):
                            for p in range(ci):
                                ui = u[bi, oi, oj, m]
                                xi = x[bi, ii, ij, p]
                                gi = xi * ui
                                g[ki, kj, p, m] += gi
    return g


def _np_conv2d_backward_b(upstream):
    np = get_numpy()
    u = upstream
    u = np.transpose(u, [0, 2, 3, 1])
    u = u.reshape(-1, u.shape[-1])
    g = u.sum(0)
    return g
