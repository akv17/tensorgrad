import os
from tensorgrad import DEVICE as _DEVICE, DTYPE as _DTYPE


def get_device():
    dispatch = {
        'cpu': _DEVICE.CPU,
        'cuda': _DEVICE.CUDA,
    }
    device = os.getenv('DEVICE', 'cpu')
    if device not in dispatch:
        msg = f'unknown device: {device}'
        raise Exception(msg)
    device = dispatch[device]
    return device


def get_dtype():
    return _DTYPE.FLOAT32


DEVICE = get_device()
DTYPE = get_dtype()
SHOW_DIFF = os.getenv('SHOW_DIFF') == '1'
