from ..const import DTYPE, DEVICE


class CPUStorage:
    _NUMPY = None

    @classmethod
    def tensor(cls, data, dtype=None):
        np = cls._get_numpy()
        dtype = cls._map_dtype(dtype)
        data = np.array(data, dtype=dtype)
        return data
    
    @classmethod
    def empty(cls, shape, dtype=None):
        np = cls._get_numpy()
        dtype = cls._map_dtype(dtype)
        data = np.empty(shape, dtype=dtype)
        return data

    @classmethod
    def zeros(cls, shape, dtype=None):
        np = cls._get_numpy()
        dtype = cls._map_dtype(dtype)
        data = np.zeros(shape, dtype=dtype)
        return data

    @classmethod
    def ones(cls, shape, dtype=None):
        np = cls._get_numpy()
        dtype = cls._map_dtype(dtype)
        data = np.ones(shape, dtype=dtype)
        return data
    
    @classmethod
    def arange(cls, n, dtype=None):
        np = cls._get_numpy()
        dtype = cls._map_dtype(dtype)
        data = np.arange(n, dtype=dtype)
        return data

    @classmethod
    def random_uniform(cls, a, b, shape, dtype=None):
        np = cls._get_numpy()
        dtype = cls._map_dtype(dtype)
        data = np.random.uniform(a, b, size=shape).astype(dtype)
        return data

    @classmethod
    def cast(cls, data, dtype):
        dtype = cls._map_dtype(dtype)
        data = data.astype(dtype)
        return data

    @classmethod
    def get_device(cls, data):
        return DEVICE.CPU
    
    @classmethod
    def get_dtype(cls, data):
        return cls._imap_dtype(data.dtype)
    
    @classmethod
    def get_shape(cls, data):
        return data.shape

    @classmethod
    def _map_dtype(cls, dtype):
        np = cls._get_numpy()
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

    @classmethod
    def _get_numpy(cls):
        if cls._NUMPY is None:
            import numpy
            cls._NUMPY = numpy
        return cls._NUMPY
