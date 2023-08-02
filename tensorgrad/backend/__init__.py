from .numpy_ import NumpyBackend
from ..const import BACKEND


def get_backend(name):
    dispatch = {
        BACKEND.NUMPY: NumpyBackend
    }
    ob = dispatch[name]
    return ob
