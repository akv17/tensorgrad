import numpy as np


def require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError('PyTorch required for testing.')


def check_tensors(a, b, tol=1e-5, show_diff=False):
    a = np.array(a.tolist()) if hasattr(a, 'tolist') else np.array(a)
    b = np.array(b.tolist()) if hasattr(b, 'tolist') else np.array(b)
    assert a.shape == b.shape, f'shape mismatch: {a.shape}, {b.shape}'
    flag = np.allclose(a, b, rtol=tol, atol=tol)
    if show_diff and not flag:
        a = a.ravel()
        b = b.ravel()
        for ai, bi in zip(a, b):
            if not np.allclose([ai], [bi], rtol=tol, atol=tol):
                msg = f'a={ai} b={bi} :: d={abs(ai - bi)}'
                print(msg)
    return flag
