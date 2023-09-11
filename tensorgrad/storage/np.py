from ..const import DTYPE


class NumpyStorage:
    np = None
    DEVICE = None

    @classmethod
    def tensor(cls, data, dtype=None):
        dtype = cls._map_dtype(dtype)
        data = cls.np.array(data, dtype=dtype)
        return data
    
    @classmethod
    def empty(cls, shape, dtype=None):
        dtype = cls._map_dtype(dtype)
        data = cls.np.empty(shape, dtype=dtype)
        return data

    @classmethod
    def zeros(cls, shape, dtype=None):
        dtype = cls._map_dtype(dtype)
        data = cls.np.zeros(shape, dtype=dtype)
        return data

    @classmethod
    def ones(cls, shape, dtype=None):
        dtype = cls._map_dtype(dtype)
        data = cls.np.ones(shape, dtype=dtype)
        return data
    
    @classmethod
    def arange(cls, n, dtype=None):
        dtype = cls._map_dtype(dtype)
        data = cls.np.arange(n, dtype=dtype)
        return data

    @classmethod
    def random_uniform(cls, a, b, shape, dtype=None):
        dtype = cls._map_dtype(dtype)
        data = cls.np.random.uniform(a, b, size=shape).astype(dtype)
        return data
    
    @classmethod
    def bernoulli(cls, p, shape, dtype=None):
        dtype = cls._map_dtype(dtype)
        data = cls.np.random.binomial(n=1, p=p, size=shape).astype(dtype)
        return data

    @classmethod
    def cast(cls, data, dtype):
        dtype = cls._map_dtype(dtype)
        data = data.astype(dtype)
        return data
    
    @classmethod
    def numpy(cls, data):
        return data

    @classmethod
    def get_device(cls, data):
        return cls.DEVICE
    
    @classmethod
    def get_dtype(cls, data):
        return cls._imap_dtype(data.dtype)
    
    @classmethod
    def get_shape(cls, data):
        return data.shape

    @classmethod
    def _map_dtype(cls, dtype):
        np = cls.np
        dtype = dtype or DTYPE.FLOAT32
        dispatch = {
            DTYPE.FLOAT32: np.float32,
            DTYPE.FLOAT64: np.float64,
            DTYPE.INT32: np.int32,
            DTYPE.INT64: np.int64,
            DTYPE.BOOL: bool,
        }
        return dispatch[dtype]
    
    @classmethod
    def _imap_dtype(cls, dtype):
        dtype = str(dtype)
        dispatch = {
            'float32': DTYPE.FLOAT32,
            'float64': DTYPE.FLOAT64,
            'int32': DTYPE.INT32,
            'int64': DTYPE.INT64,
            'bool': DTYPE.BOOL,
        }
        return dispatch[dtype]
