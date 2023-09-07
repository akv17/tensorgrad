from .cpu import CPUStorage as _NumpyStorage
from ..const import DEVICE

class CUDAStorage(_NumpyStorage):

    @classmethod
    def get_device(cls, data):
        return DEVICE.CUDA

    @classmethod
    def _get_numpy(cls):
        if cls._NUMPY is None:
            import cupy
            cls._NUMPY = cupy
        return cls._NUMPY
