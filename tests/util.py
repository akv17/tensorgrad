import itertools

import numpy as np


def require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError('PyTorch required for testing.')


def check_tensors(a, b, tol=1e-5, show_diff=False):
    a = np.array(a)
    b = np.array(b)
    assert a.shape == b.shape, 'shape mismatch'
    flag = np.allclose(a, b, rtol=tol, atol=tol)
    if show_diff and not flag:
        a = a.ravel()
        b = b.ravel()
        for ai, bi in zip(a, b):
            if not np.allclose([ai], [bi], rtol=tol, atol=tol):
                msg = f'a={ai} b={bi} :: d={abs(ai - bi)}'
                print(msg)
    return flag


def generate_cases(*args):
    cases = list(itertools.product(*args))
    return cases
