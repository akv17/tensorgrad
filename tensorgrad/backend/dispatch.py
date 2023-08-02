from .numpy_ import NumpyBackend
from ..const import BACKEND

_DISPATCH = {
    BACKEND.NUMPY: NumpyBackend
}


class BackendDispatch:

    @classmethod
    def get(cls, name):
        return _DISPATCH[name]
