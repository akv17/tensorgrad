from .numpy_ import NumpyBackend
from ..const import BACKEND

_DISPATCH = {
    BACKEND.NUMPY: NumpyBackend
}


class BackendDispatch:
    DEFAULT_BACKEND = BACKEND.NUMPY

    @classmethod
    def get(cls, name=None):
        name = name or cls.DEFAULT_BACKEND
        return _DISPATCH[name]

    def get_default(cls):
        return _DISPATCH[cls.DEFAULT_BACKEND]
