import math

from ..storage import StorageDispatch


def zeros(tensor):
    storage = StorageDispatch.get(tensor.device)
    data = storage.zeros(tensor.shape, dtype=tensor.dtype)
    tensor.data = data


def ones(tensor):
    storage = StorageDispatch.get(tensor.device)
    data = storage.ones(tensor.shape, dtype=tensor.dtype)
    tensor.data = data


def uniform_fan_in(tensor):
    fi, _ = _calculate_fan_in_fan_out(tensor.shape)
    hi = math.sqrt(1 / fi)
    lo = -hi
    storage = StorageDispatch.get(tensor.device)
    data = storage.random_uniform(lo, hi, shape=tensor.shape, dtype=tensor.dtype)
    tensor.data = data


def _calculate_fan_in_fan_out(shape):
    ndim = len(shape)
    if ndim == 1:
        fo, fi = None, shape[0]
    elif ndim == 2:
        fo, fi = shape
    # conv2d kernel of shape [co, ci, kh, kw]
    elif ndim == 4:
        co, ci, kh, kw = shape
        fo = co * kh * kw
        fi = ci * kh * kw
    else:
        msg = f'cannot calculate fan_in and fan_out for shape: {shape}'
        raise Exception(msg)
    return fi, fo
