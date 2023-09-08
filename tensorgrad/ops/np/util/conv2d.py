import math


def conv2d_compute_output_size(ih, iw, kh, kw, sh, sw):
    oh = math.floor((ih - kh) / sh + 1)
    ow = math.floor((iw - kw) / sw + 1)
    return oh, ow


def conv2d_extract_windows(np, x, oh, ow, kh, kw, sh, sw):
    isz = x.itemsize
    bs = x.shape[0]
    ci = x.shape[1]
    bss, cis, ihs, _ = np.array(x.strides) // isz
    bss = bss.item()
    cis = cis.item()
    ihs = ihs.item()
    shape = (bs, ci, oh, ow, kh, kw)
    strides = np.array([bss, cis, ihs * sh, sw, ihs, 1]) * isz
    strides = strides.tolist()
    w = np.lib.stride_tricks.as_strided(x, shape, strides)
    return w


def conv2d_dilate(np, x, dh, dw):
    b, c, h, w = x.shape
    h += (dh * (h - 1))
    w += (dw * (w - 1))
    o = np.zeros((b, c, h, w), dtype=x.dtype)
    o[..., ::dh+1, ::dw+1] = x
    return o
