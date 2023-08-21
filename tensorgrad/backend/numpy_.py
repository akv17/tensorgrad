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
    def random_uniform(cls, a, b, shape, dtype=None):
        dtype = dtype or DTYPE.FLOAT32
        np = cls._get_numpy()
        data = np.random.uniform(a, b, size=shape)
        ob = NumpyTensor(np=np, data=data)
        return ob

    @classmethod
    def map_dtype(cls, dtype):
        np = cls._get_numpy()
        dispatch = {
            None: np.float32,
            DTYPE.FLOAT32: np.float32,
            DTYPE.FLOAT64: np.float64,
            DTYPE.INT32: np.int32,
            DTYPE.INT64: np.int64,
            DTYPE.BOOL: bool,
        }
        return dispatch[dtype]
    
    @classmethod
    def imap_dtype(cls, dtype):
        np = cls._get_numpy()
        dispatch = {
            np.float32: DTYPE.FLOAT32,
            np.float64: DTYPE.FLOAT64,
            np.int32: DTYPE.INT32,
            np.int64: DTYPE.INT64,
            bool: DTYPE.BOOL,
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

    @property
    def dtype(self):
        value = NumpyBackend.imap_dtype(self.data.dtype)
        return value

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return f'NumpyTensor({repr(self.data)})'

    def __getitem__(self, slice_):
        out = self._new(self.data[slice_])
        return out

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
    
    def __gt__(self, value):
        out = self._new(self.data > value)
        return out
    
    def __ge__(self, value):
        out = self._new(self.data >= value)
        return out
    
    def __lt__(self, value):
        out = self._new(self.data < value)
        return out
    
    def __le__(self, value):
        out = self._new(self.data <= value)
        return out
    
    def __eq__(self, value):
        out = self._new(self.data == value)
        return out
    
    def __ne__(self, value):
        out = self._new(self.data != value)
        return out
    
    def exp(self):
        out = self._new(self.np.exp(self.data))
        return out

    def log(self):
        out = self._new(self.np.log(self.data))
        return out
    
    def sqrt(self):
        out = self._new(self.np.sqrt(self.data))
        return out

    def sum(self, dim=None):
        out = self._new(self.np.sum(self.data, dim))
        return out
    
    def mean(self, dim=None):
        out = self._new(self.np.mean(self.data, dim))
        return out

    def matmul(self, other):
        out = self._new(self.np.matmul(self.data, other.data))
        return out
    
    def conv2d(self, kernel, stride=None, padding=None):
        # x: [b, c_in, h, w]
        # k: [c_out, c_in, h, w]

        import math
        import numpy as np
        x = self.data
        k = kernel.data
        
        b = x.shape[0]
        c_in = x.shape[1]
        c_out = k.shape[0]
        h_in, w_in = x.shape[-2:]
        h_k, w_k = k.shape[-2:]
        stride = stride or (1, 1)
        h_s, w_s = stride
        padding = padding or (0, 0)
        h_p, w_p = padding
        dilation = (1, 1)
        h_d, w_d = dilation
        h_out = math.floor(((h_in + 2 * h_p - h_d * (h_k - 1) - 1) / h_s) + 1)
        w_out = math.floor(((w_in + 2 * w_p - w_d * (w_k - 1) - 1) / w_s) + 1)

        res = np.zeros((b, c_out, h_out, w_out))
        x_pad = x
        if padding != (0, 0):
            x_pad_shape = (b, c_in, h_in + h_p * 2, w_in + w_p * 2)
            x_pad = np.zeros(x_pad_shape, dtype=x.dtype)
            x_pad[..., h_p:x_pad.shape[2]-h_p, w_p:x_pad.shape[3]-w_p] = x
        hh_k = math.floor(h_k / 2)
        hw_k = math.floor(w_k / 2)
        h_start = hh_k
        h_end = x_pad.shape[2] - hh_k - 1
        w_start = hw_k
        w_end = x_pad.shape[3] - hw_k - 1
        for bi in range(b):
            i_out = 0
            for i_in in range(h_start, h_end + 1, h_s):
                j_out = 0
                for j_in in range(w_start, w_end + 1, w_s):
                    xp = x_pad[bi, :, i_in-hh_k:i_in+hh_k+1, j_in-hw_k:j_in+hw_k+1]
                    for c in range(c_out):
                        res[bi, c, i_out, j_out] = (k[c] * xp).sum()
                    j_out += 1
                i_out += 1
        
        res = self._new(res)
        return res


    def fill(self, mask, value):
        out = self.data.copy()
        out[mask.data] = value
        out = self._new(out)
        return out
    
    def ifill(self, mask, value):
        out = self.data.copy()
        out[~mask.data] = value
        out = self._new(out)
        return out

    def fill_diagonal(self, other):
        out = self.data.copy()
        self.np.fill_diagonal(out, other.data)
        out = self._new(out)
        return out

    def fill_diagonal2d(self, other):
        out = self.data.copy()
        dim_last = out.shape[-1]
        dim_prelast = out.shape[-2]
        out[..., range(dim_prelast), range(dim_last)] = other.data
        out = self._new(out)
        return out

    def put(self, slice_, other):
        out = self.data.copy()
        out[slice_] = other.data
        out = self._new(out)
        return out

    def reshape(self, *shape):
        out = self._new(self.data.reshape(*shape))
        return out

    def permute(self, shape):
        out = self._new(self.np.transpose(self.data, shape))
        return out

    def squeeze(self, dim):
        out = self._new(self.np.squeeze(self.data, dim))
        return out

    def unsqueeze(self, dim):
        out = self._new(self.np.expand_dims(self.data, dim))
        return out

    def numpy(self):
        return self.data.copy()

    def tolist(self):
        return self.data.tolist()

    def copy(self):
        return self._new(self.data)

    def zeros_like(self):
        return self._new(self.np.zeros_like(self.data))
    
    def ones_like(self):
        return self._new(self.np.ones_like(self.data))

    def _new(self, data):
        return type(self)(np=self.np, data=data)

    def _maybe_wrap_constant(self, other):
        if isinstance(other, (int, float)):
            other = namedtuple('C', ('data',))(other)
        return other
