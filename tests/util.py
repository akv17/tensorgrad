import itertools

import numpy as np


def require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError('PyTorch required for testing.')


def check_tensors(a, b, tol=1e-5):
    a = np.array(a)
    b = np.array(b)
    flag = np.allclose(a, b, rtol=tol, atol=tol)
    return flag


def generate_cases(*args):
    cases = list(itertools.product(*args))
    return cases
