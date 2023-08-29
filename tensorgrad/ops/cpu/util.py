_NUMPY = None


def get_numpy():
    global _NUMPY
    if _NUMPY is None:
        import numpy
        _NUMPY = numpy
    return _NUMPY
