import math

import numpy as np


def conv2d_compute_output_size(ih, iw, kh, kw, sh, sw):
    oh = math.floor((ih - kh) / sh + 1)
    ow = math.floor((iw - kw) / sw + 1)
    return oh, ow


def conv2d_extract_windows(x, oh, ow, kh, kw, sh, sw):
    isz = x.itemsize
    bs = x.shape[0]
    ci = x.shape[1]
    bss, cis, ihs, _ = np.array(x.strides) // isz
    shape = (bs, ci, oh, ow, kh, kw)
    strides = np.array([bss, cis, ihs * sh, sw, ihs, 1]) * isz
    w = np.lib.stride_tricks.as_strided(x, shape, strides)
    return w


def conv2d_dilate(x, dh, dw, ah=2, aw=3):
    idx = np.arange(x.shape[ah] - 1) + 1
    idx = np.repeat(idx, dh)
    x = np.insert(x, idx, 0, axis=ah)
    idx = np.arange(x.shape[aw] - 1) + 1
    idx = np.repeat(idx, dw)
    x = np.insert(x, idx, 0, axis=aw)
    return x
