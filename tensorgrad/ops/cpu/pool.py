import warnings

from .util.np import NumpyProvider
from .util.conv2d import conv2d_compute_output_size, conv2d_extract_windows, conv2d_dilate
from ..stubs import BaseOp
from ..dispatch import OpDispatch
from ...const import OP, DEVICE


@OpDispatch.register(OP.MAX_POOL2D, DEVICE.CPU)
class MaxPool2D(BaseOp, NumpyProvider):

    def __init__(self, x, *, kernel_size, stride=None, padding=None):
        self.x = x
        self.kernel_size = kernel_size
        self.stride = stride or self.kernel_size
        self.padding = padding or (0, 0)
        
        self._use_fast_impl = self._decide_use_fast_impl()
        if not self._use_fast_impl:
            msg = (
                'MaxPool2D will use slow implementation because input cannot be evenly tiled with given parameters. '
                'To use fast implementation consider choosing square kernel with stride of the same size and padding such that total input size is divisible by kernel size.'
            )
            warnings.warn(msg)
            self._jit_backward_slow = self._jit_compile_backward_slow()
            self._forward = self._forward_slow
            self._backward = self._backward_slow
        else:
            self._forward = self._forward_fast
            self._backward = self._backward_fast
    
    def forward(self):
        o = self._forward()
        self.out = self.x.from_data(o)
        return self.out

    def backward(self):
        if not self.x.requires_grad:
            return
        g = self._backward()
        self.x.grad += g

    def _forward_fast(self):
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
        oh = ih // sh
        ow = iw // sw

        # w[..., i, :, j, :] -> window `i,j` of input which corresponds to pooled pixel 'i,j' of output.
        w = x.reshape(bs, ci, oh, kh, ow, kw)
        # pool each window over `kh,kw` dims.
        o = w.max((3, 5))
        return o
    
    def _backward_fast(self):
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
        oh = ih // sh
        ow = iw // sw
        
        # same windows as in forward.
        w = x.reshape(bs, ci, oh, kh, ow, kw)
        o = self.out.data
        u = self.out.grad
        # o[i, :, j, :] -> output pixel `i,j` which will be broadcasted to window `i,j` of input.
        o = o.reshape(bs, ci, oh, 1, ow, 1)
        # u[i, :, j, :] -> upstream pixel `i,j` which will be broadcasted to window `i,j` of input.
        u = u.reshape(bs, ci, oh, 1, ow, 1)
        # mask of pooled pixels into input windows.
        m = w == o
        m = m.astype('float32')
        g = u * m
        g = g.reshape(x.shape)

        if ph > 0 or pw > 0:
            g = g[..., ph:-ph, pw:-pw]
        return g
    
    def _forward_slow(self):
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
        self._mask = mask
        o = w.max(-1)
        return o

    def _backward_slow(self):
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
        # could not come up with pure numpy implementation thus isung numba.jit to speedup explicit scalar loop.
        g = self._jit_backward_slow(
            g,
            u,
            self._mask,
            bs, ci, oh, ow, kh, kw,sh,sw
        )

        if ph > 0 or pw > 0:
            g = g[..., ph:-ph, pw:-pw]
        return g
    
    def _decide_use_fast_impl(self):
        ph, pw = self.padding
        sh, sw = self.stride
        ih, iw = self.x.shape[-2:]
        kh, kw = self.kernel_size
        ih += (ph * 2)
        iw += (pw * 2)
        kernel_flag = kh == kw == sh == sw
        size_flag = (ih % kh == 0) and (iw % kw == 0)
        flag = kernel_flag and size_flag
        return flag

    def _jit_compile_backward_slow(self):
        try:
            import numba
        except ImportError:
            msg = 'cannot import numba: numba is required to jit compile slow backward kernel of MaxPool2D.'
            raise ImportError(msg)
        
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
        
        return _jit_max_pool2d_backward


@OpDispatch.register(OP.AVG_POOL2D, DEVICE.CPU)
class AvgPool2D(BaseOp, NumpyProvider):

    def __init__(self, x, *, kernel_size, stride=None, padding=None):
        self.x = x
        self.kernel_size = kernel_size
        self.stride = stride or self.kernel_size
        self.padding = padding or (0, 0)
    
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
