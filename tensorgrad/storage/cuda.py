from .cpu import CPUStorage as _NumpyStorage
from ..const import DEVICE

_CUPY = None


class CUDAStorage(_NumpyStorage):

    @classmethod
    def get_device(cls, data):
        return DEVICE.CUDA

    @classmethod
    def _get_numpy(cls):
        global _CUPY
        if _CUPY is None:
            import cupy
            _CUPY = cupy
        return _CUPY
