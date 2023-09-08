from ..const import DEVICE
from .cpu import CPUStorage
from .cuda import CUDAStorage


class StorageDispatch:
    _DISPATCH = {
        DEVICE.CPU: CPUStorage,
        DEVICE.CUDA: CUDAStorage,
    }

    @classmethod
    def get(cls, device):
        device = device or DEVICE.CPU
        ob = cls._DISPATCH[device]
        return ob
