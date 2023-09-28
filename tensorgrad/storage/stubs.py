from abc import ABC, abstractmethod


class Storage(ABC):
    DEVICE = None

    @classmethod
    @abstractmethod
    def tensor(cls, data, dtype=None):
        """Construct multi-dimensional array from data of given dtype. Data may be a native python list or a numpy array"""
    
    @classmethod
    @abstractmethod
    def empty(cls, shape, dtype=None):
        """Construct empty array of given shape and dtype"""

    @classmethod
    @abstractmethod
    def zeros(cls, shape, dtype=None):
        """Construct array of zeroes of given shape and dtype"""

    @classmethod
    @abstractmethod
    def ones(cls, shape, dtype=None):
        """Construct array of ones of given shape and dtype"""
    
    @classmethod
    @abstractmethod
    def arange(cls, n, dtype=None):
        """Construct range array of given size and dtype"""

    @classmethod
    @abstractmethod
    def random_uniform(cls, a, b, shape, dtype=None):
        """Construct array of random uniform distribution with given parameters, shape and size"""
    
    @classmethod
    @abstractmethod
    def random_normal(cls, shape, mu=0.0, sigma=1.0, dtype=None):
        """Construct array of random normal distribution with given parameters, shape and size"""
    
    @classmethod
    @abstractmethod
    def random_randint(cls, low, high, shape, dtype=None):
        """Construct array of random integers with given parameters, shape and size"""
    
    @classmethod
    @abstractmethod
    def bernoulli(cls, p, shape, dtype=None):
        """Construct array of random bernoulli distribution with given parameters, shape and size"""

    @classmethod
    @abstractmethod
    def cast(cls, data, dtype):
        """Cast data to given dtype"""

    @classmethod
    @abstractmethod
    def numpy(cls, data):
        """Represent data as a numpy array"""

    @classmethod
    @abstractmethod
    def get_device(cls, data): 
        """Return name of the storage's device"""
    
    @classmethod
    @abstractmethod
    def get_dtype(cls, data):
        """Return dtype of data as tensorgrad.DTYPE"""
    
    @classmethod
    @abstractmethod
    def get_shape(cls, data):
        """Return shape of data as a tuple"""
