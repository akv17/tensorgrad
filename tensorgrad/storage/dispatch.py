from ..const import DEVICE
from .cpu import CPUStorage


class StorageDispatch:
    _DISPATCH = {
        DEVICE.CPU: CPUStorage
    }

    @classmethod
    def get(cls, device):
        device = device or DEVICE.CPU
        ob = cls._DISPATCH[device]
        return ob
