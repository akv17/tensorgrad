_NUMPY = None
_NUMBA = None


def get_numpy():
    global _NUMPY
    if _NUMPY is None:
        import numpy
        _NUMPY = numpy
    return _NUMPY


def get_numba():
    global _NUMBA
    if _NUMBA is None:
        import numba
        _NUMBA = numba
    return _NUMBA
