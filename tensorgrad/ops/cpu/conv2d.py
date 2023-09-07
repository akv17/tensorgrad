from .util.np import NumpyProvider
from .util.conv2d import conv2d_compute_output_size, conv2d_extract_windows, conv2d_dilate
from ..stubs import BaseOp
from ..dispatch import OpDispatch
from ...const import OP, DEVICE


@OpDispatch.register(OP.CONV2D, DEVICE.CPU)
class Conv2D(BaseOp, NumpyProvider):
    # this op heavily uses np.einsum as it's significantly faster than 2d matmuls with reshapes.
    # also it turns out that multidim tensor matmuls are ridiculously slow.

    def __init__(self, x, kernel, bias=None, *, stride=None, padding=None):
        self.out = None
        self.x = x
        self.kernel = kernel
        self.bias = bias
        self.stride = stride or (1, 1)
        self.padding = padding or (0, 0)

    def forward(self):
        bias = self.bias.data if self.bias is not None else None
        data = self._forward(
            x=self.x.data,
            k=self.kernel.data,
            b=bias,
            sh=self.stride[0],
            sw=self.stride[1],
            ph=self.padding[0],
            pw=self.padding[1],
        )
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            g = self._backward_x(
                u=self.out.grad,
                x=self.x.data,
                k=self.kernel.data,
                sh=self.stride[0],
                sw=self.stride[1],
                ph=self.padding[0],
                pw=self.padding[1],
            )
            self.x.grad += g
        
        if self.kernel.requires_grad:
            g = self._backward_k(
                u=self.out.grad,
                x=self.x.data,
                k=self.kernel.data,
                sh=self.stride[0],
                sw=self.stride[1],
                ph=self.padding[0],
                pw=self.padding[1],
            )
            self.kernel.grad += g
        
        if self.bias is not None and self.bias.requires_grad:
            g = self._backward_b(self.out.grad)
            self.bias.grad += g

    def _forward(self, x, k, b, sh, sw, ph, pw):
        np = self.np
        
        if ph > 0 or pw > 0:
            x = np.pad(x, [(0, 0), (0, 0), (ph, ph), (pw, pw)])
        
        ih, iw = x.shape[-2:]
        kh, kw = k.shape[-2:]
        oh, ow = conv2d_compute_output_size(ih, iw, kh, kw, sh, sw)
        co = k.shape[0]
        b = b if b is not None else np.zeros((co,), dtype=k.dtype)
        
        # windows into input of shape: B x IC x OH x OW x KH x KW.
        # basically this is a collection of all windows to which kernel is applied.
        # kernel is multiplied and reduced with each such window.
        w = conv2d_extract_windows(np, x, oh, ow, kh, kw, sh, sw)
        o = np.einsum('bihwkl,oikl->bohw', w, k, optimize=True)
        o += b.reshape(1, -1, 1, 1)
        return o

    def _backward_x(self, u, x, k, sh, sw, ph, pw):
        np = self.np
        
        ih, iw = x.shape[-2:]
        ihp = ih + (ph * 2)
        iwp = iw + (pw * 2)
        kh, kw = k.shape[-2:]

        # this is hacky but this allows to compute the grad in terms of convolution.
        # we dilate upstream according to forward stride and then pad upstream according to forward padding.
        # next we need to rotate the kernel along its height and width.
        # finally grad is computed effectively as conv2d(x=upstream_modified, k=kernel_modified).
        udh = sh - 1
        udw = sw - 1
        _u = conv2d_dilate(np, u, udh, udw)
        uph = kh - 1
        upw = kw - 1
        _u = np.pad(_u, [(0, 0), (0, 0), (uph, uph), (upw, upw)])
        _k = np.rot90(k, k=2, axes=(2, 3))

        # windows into upstream of shape: B x CO x IH x IW x KH x KW. 
        # note that IH and IW are padded according to padding applied in forward pass.
        # kernel is multiplied and reduced with each such window.
        # finally we cancel padding by slicing off pads.
        w = conv2d_extract_windows(np, _u, ihp, iwp, kh, kw, 1, 1)
        g = np.einsum('bohwkl,oikl->bihw', w, _k, optimize=True)
        
        if ph != 0:
            g = g[..., ph:-ph, :]
        if pw != 0:
            g = g[..., pw:-pw]
        return g
    
    def _backward_k(self, u, x, k, sh, sw, ph, pw):
        np = self.np
        
        if ph != 0 or pw != 0:
            x = np.pad(x, [(0, 0), (0, 0), (ph, ph), (pw, pw)])
        
        ih, iw = x.shape[-2:]
        kh, kw = k.shape[-2:]
        oh, ow = conv2d_compute_output_size(ih, iw, kh, kw, sh, sw)

        # windows into input of same shape as in forward.
        # via einsum each such window is multiplied with corresponding pixel in the upstream.
        # then the result is reduced over batch, `kh` and `kw` dims of each input window.
        w = conv2d_extract_windows(np, x, oh, ow, kh, kw, sh, sw)
        g = np.einsum('bihwkl,bohw->oikl', w, u, optimize=True)
        return g

    def _backward_b(self, u):
        g = u.sum((0, 2, 3))
        return g
