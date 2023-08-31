import numba
import numpy as np

from .util import get_numpy
from .util.conv2d import conv2d_compute_output_size, conv2d_extract_windows, conv2d_dilate
from ..stubs import BaseOp
from ..dispatch import OpDispatch
from ...const import OP, DEVICE


@OpDispatch.register(OP.MAX_POOL2D, DEVICE.CPU)
class MaxPool2D(BaseOp):

    def __init__(self, x, *, kernel_size, stride=None, padding=None):
        self.x = x
        self.kernel_size = kernel_size
        self.stride = stride or self.kernel_size
        self.padding = padding or (0, 0)
        self.np = get_numpy()
    
    def forward(self):
        np = self.np

        x = self.x.data
        ph, pw = self.padding
        if ph > 0 or pw > 0:
            x = np.pad(x, [(0, 0), (0, 0), (ph, ph), (pw, pw)], constant_values=-np.inf)

        ih, iw = x.shape[-2:]
        bs = x.shape[0]
        ci = x.shape[1]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = conv2d_compute_output_size(ih, iw, kh, kw, sh, sw)
        
        w = conv2d_extract_windows(x, oh, ow, kh, kw, sh, sw)
        w = w.reshape(bs, ci, oh, ow, kh * kw)
        mask = np.argmax(w, -1)
        self.mask = mask
        o = w.max(-1)
        self.out = self.x.from_data(o)
        return self.out
    
    def backward(self):
        if not self.x.requires_grad:
            return
        np = self.np

        x = self.x.data
        ph, pw = self.padding
        if ph > 0 or pw > 0:
            x = np.pad(x, [(0, 0), (0, 0), (ph, ph), (pw, pw)], constant_values=-np.inf)
        
        ih, iw = x.shape[-2:]
        bs = x.shape[0]
        ci = x.shape[1]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = conv2d_compute_output_size(ih, iw, kh, kw, sh, sw)

        u = self.out.grad
        g = np.zeros_like(x)
        # could not come up with pure numpy implementation thus isung numba.jit.
        g = _jit_max_pool2d_backward(
            g,
            u,
            self.mask,
            bs, ci, oh, ow, kh, kw,sh,sw
        )

        if ph > 0 or pw > 0:
            g = g[..., ph:-ph, pw:-pw]
        self.x.grad += g


@numba.jit(nopython=True)
def _jit_max_pool2d_backward(g, u, m, bs, ci, oh, ow, kh, kw, sh, sw):
    for b in range(bs):
        for c in range(ci):
            for oi in range(oh):
                for oj in range(ow):
                    # mask stores flat pooled index into kernel dims.
                    # need to unravel it to 2d index into H, W input dims.
                    xi_flat = m[b, c, oi, oj]
                    xi = xi_flat // kh
                    xj =  xi_flat % kw
                    xi += oi * sh
                    xj += oj * sw
                    ui = u[b, c, oi, oj]
                    g[b, c, xi, xj] += ui
    return g


@OpDispatch.register(OP.AVG_POOL2D, DEVICE.CPU)
class AvgPool2D(BaseOp):

    def __init__(self, x, *, kernel_size, stride=None, padding=None):
        self.x = x
        self.kernel_size = kernel_size
        self.stride = stride or self.kernel_size
        self.padding = padding or (0, 0)
        self.np = get_numpy()
    
    def forward(self):
        np = self.np

        x = self.x.data
        ph, pw = self.padding
        if ph > 0 or pw > 0:
            x = np.pad(x, [(0, 0), (0, 0), (ph, ph), (pw, pw)], constant_values=0.0)

        ih, iw = x.shape[-2:]
        bs = x.shape[0]
        ci = x.shape[1]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = conv2d_compute_output_size(ih, iw, kh, kw, sh, sw)
        
        w = conv2d_extract_windows(x, oh, ow, kh, kw, sh, sw)
        w = w.reshape(bs, ci, oh, ow, kh * kw)
        o = w.mean(-1)
        self.out = self.x.from_data(o)
        return self.out
    
    def backward(self):
        if not self.x.requires_grad:
            return
        
        np = self.np

        ph, pw = self.padding
        sh, sw = self.stride

        x = self.x
        u = self.out.grad
        kh, kw = self.kernel_size
        k = np.tile(1 / (kh * kw), [kh, kw]).astype(u.dtype)

        ih, iw = x.shape[-2:]
        ihp = ih + (ph * 2)
        iwp = iw + (pw * 2)

        # using the same trick as in conv2d.backward_x.
        # this allows to compute gradient as conv2d(upstream, kernel).
        
        udh = sh - 1
        udw = sw - 1
        _u = conv2d_dilate(u, udh, udw)
        uph = kh - 1
        upw = kw - 1
        _u = np.pad(_u, [(0, 0), (0, 0), (uph, uph), (upw, upw)])
        _k = np.rot90(k, k=2, axes=(0, 1))

        w = conv2d_extract_windows(_u, ihp, iwp, kh, kw, 1, 1)
        g = np.einsum('bchwkl,kl->bchw', w, _k, optimize=True)
        
        if ph != 0:
            g = g[..., ph:-ph, :]
        if pw != 0:
            g = g[..., pw:-pw]
        
        self.x.grad += g
