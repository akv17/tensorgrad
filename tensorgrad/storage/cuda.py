from .np import NumpyStorage
from ..const import DEVICE
from ..util.lazy import LazyImport


class CUDAStorage(NumpyStorage):
    np = LazyImport('cupy', err_msg='Storage of CUDA device is not available')
    DEVICE = DEVICE.CUDA

    @classmethod
    def numpy(cls, data):
        data = cls.np.asnumpy(data)
        return data
