from .np import NumpyStorage
from ..const import DEVICE
from ..util.lazy import LazyImport


class CPUStorage(NumpyStorage):
    np = LazyImport('numpy', err_msg='Storage of CPU device is not available')
    DEVICE = DEVICE.CPU
