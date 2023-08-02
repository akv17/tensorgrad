from collections import namedtuple

from ..const import DTYPE


class NumpyBackend:
    _NUMPY = None

    @classmethod
    def tensor(cls, data, dtype=None):
        if hasattr(data, 'numpy'):
            buffer = data.numpy()
        else:
            buffer = data
        np = cls._get_numpy()
        dtype = cls.map_dtype(dtype)
        data = np.array(buffer, dtype=dtype)
        tensor = NumpyTensor(np=np, data=data)
        return tensor

    @classmethod
    def zeros(cls, shape, dtype=None):
        np = cls._get_numpy()
        dtype = cls.map_dtype(dtype)
        data = np.zeros(shape, dtype=dtype)
        ob = NumpyTensor(np=np, data=data)
        return ob

    @classmethod
    def ones(cls, shape, dtype=None):
        np = cls._get_numpy()
        dtype = NumpyBackend.map_dtype(dtype)
        data = np.ones(shape, dtype=dtype)
        ob = NumpyTensor(np=np, data=data)
        return ob

    @classmethod
    def map_dtype(cls, dtype):
        np = cls._get_numpy()
        dispatch = {
            DTYPE.FLOAT32: np.float32,
            DTYPE.FLOAT64: np.float64,
            DTYPE.INT32: np.int32,
            DTYPE.INT64: np.int64,
            DTYPE.BOOL: bool,
        }
        return dispatch[dtype]

    @classmethod
    def _get_numpy(cls):
        if cls._NUMPY is None:
            import numpy
            cls._NUMPY = numpy
        return cls._NUMPY


class NumpyTensor:

    def __init__(self, np, data):
        self.np = np
        self.data = data
        self.dtype = self.data.dtype

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return f'NumpyTensor({repr(self.data)})'

    def __add__(self, other):
        other = self._maybe_wrap_constant(other)
        out = self._new(self.data + other.data)
        return out
    
    __radd__ = __add__

    def __sub__(self, other):
        other = self._maybe_wrap_constant(other)
        out = self._new(self.data - other.data)
        return out
    
    def __rsub__(self, other):
        other = self._maybe_wrap_constant(other)
        out = self._new(other.data - self.data)
        return out

    def __mul__(self, other):
        other = self._maybe_wrap_constant(other)
        out = self._new(self.data * other.data)
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._maybe_wrap_constant(other)
        out = self._new(self.data / other.data)
        return out
    
    def __rtruediv__(self, other):
        other = self._maybe_wrap_constant(other)
        out = self._new(other.data / self.data)
        return out

    def __pow__(self, value):
        out = self._new(self.data ** value)
        return out

    def __neg__(self):
        out = self._new(-1.0 * self.data)
        return out
    
    def exp(self):
        out = self._new(self.np.exp(self.data))
        return out

    def log(self):
        out = self._new(self.np.log(self.data))
        return out

    def sum(self, dim=None):
        out = self._new(self.np.sum(self.data, dim))
        return out
    
    def mean(self, dim=None):
        out = self._new(self.np.mean(self.data, dim))
        return out

    def numpy(self):
        return self.data.copy()

    def tolist(self):
        return self.data.tolist()

    def copy(self):
        return self._new(self.data)

    def _new(self, data):
        return type(self)(np=self.np, data=data)

    def _maybe_wrap_constant(self, other):
        if isinstance(other, (int, float)):
            other = namedtuple('C', ('data',))(other)
        return other
