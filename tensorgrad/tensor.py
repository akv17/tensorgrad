from .const import OP, DTYPE
from .storage import StorageDispatch
from .ops import OpDispatch
from .ctx import is_grad_enabled


class Tensor:
    """
    Implementation of a multi-dimensional array of specific data type on a specific device.  
    Provides core arithmetic, indexing and shape operations, factory functions and utilities.  

    Arithmetic and shape operations preserve data type and device placement.  
    Factory functions accept desired data type and device as arguments.  
    Utilities mainly provide means to cast data types or move data to another device.  

    Supported data types:  
    - `tensorgrad.DTYPE.FLOAT32`  
    - `tensorgrad.DTYPE.FLOAT64`  
    - `tensorgrad.DTYPE.INT32`  
    - `tensorgrad.DTYPE.INT64`  
    - `tensorgrad.DTYPE.BOOL`  

    Supported devices:  
    - `tensorgrad.DEVICE.CPU`  
    - `tensorgrad.DEVICE.CUDA`  

    All arithmetic, indexing and shape operations are automatically recorded into computational graph by autograd engine.  
    Each node in the graph is a tensor being result of some operation.  
    Such graph may then be traversed forward to compute output or backward to perform reverse-mode differentiation.  
    Graph traversal may be started from any node.  
    Tensor may be excluded from the graph by setting `requires_grad` to `False` or calling `detach()`.  
    """

    @classmethod
    def empty(cls, *shape, dtype=None, device=None, requires_grad=True):
        """Construct empty tensor with given spec"""
        tensor = cls._factory('empty', shape=shape, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor
    
    @classmethod
    def zeros(cls, *shape, dtype=None, device=None, requires_grad=True):
        """Construct tensor of zeros with given spec"""
        tensor = cls._factory('zeros', shape=shape, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor
    
    @classmethod
    def ones(cls, *shape, dtype=None, device=None, requires_grad=True):
        """Construct tensor of ones with given spec"""
        tensor = cls._factory('ones', shape=shape, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor
    
    @classmethod
    def arange(cls, n, dtype=None, device=None, requires_grad=True):
        """Construct range tensor with given spec"""
        dtype = dtype or DTYPE.INT64
        storage = StorageDispatch.get(device)
        data = storage.arange(n, dtype=dtype)
        tensor = cls(data=data, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor
    
    @classmethod
    def rand(cls, *shape, dtype=None, device=None, requires_grad=True):
        """Construct tensor from random uniform distribution [0, 1) with given spec"""
        kwargs = {'a': 0.0, 'b': 1.0}
        tensor = cls._factory('random_uniform', shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, **kwargs)
        return tensor
    
    @classmethod
    def randn(cls, *shape, mu=0.0, sigma=1.0, dtype=None, device=None, requires_grad=True):
        """Construct tensor from random normal distribution with given spec and mu and sigma"""
        kwargs = {'mu': mu, 'sigma': sigma}
        tensor = cls._factory('random_normal', shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, **kwargs)
        return tensor
    
    @classmethod
    def randint(cls, low, high, shape, dtype=None, device=None, requires_grad=True):
        """Construct tensor of random integers in range [low, high) with given spec"""
        dtype = dtype or DTYPE.INT64
        kwargs = {'low': low, 'high': high, 'shape': shape}
        tensor = cls._factory('random_randint', dtype=dtype, device=device, requires_grad=requires_grad, **kwargs)
        return tensor
    
    @classmethod
    def bernoulli(cls, p, shape, dtype=None, device=None, requires_grad=True):
        """Construct tensor from random bernoulli distribution with given spec and parameter `p`"""
        kwargs = {'p': p}
        tensor = cls._factory('bernoulli', shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, **kwargs)
        return tensor

    @classmethod
    def _factory(cls, method, shape, dtype, device, requires_grad, **kwargs):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        storage = StorageDispatch.get(device)
        data = getattr(storage, method)(shape=shape, dtype=dtype, **kwargs)
        tensor = cls(data=data, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor

    def __init__(
        self,
        data,
        dtype=None,
        device=None,
        requires_grad=True,
        name=None,
    ):
        """Construct tensor explicitly from data with given spec"""
        self.name = name
        self.requires_grad = requires_grad if is_grad_enabled() else False
        
        self._storage = StorageDispatch.get(device)
        self.data = self._storage.tensor(data, dtype=dtype)
        self.grad = self._storage.zeros(self.data.shape, dtype=dtype)
        
        self._children = ()
        self._op = None

    @property
    def dtype(self):
        """Data type of each element"""
        return self._storage.get_dtype(self.data)

    @property
    def device(self):
        """Device the tensor is allocated on"""
        return self._storage.get_device(self.data)

    @property
    def shape(self):
        """Shape of the tensor"""
        return self._storage.get_shape(self.data)
    
    @property
    def ndim(self):
        """Number of tensor dimensions"""
        return len(self.shape)
    
    def numel(self):
        """Total number of elements in the tensor"""
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def _backward(self):
        return self._op.backward if self._op is not None else lambda: None
    
    def __repr__(self):
        return f'Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device})'

    def __getitem__(self, slice_):
        """Retreive tensor slice by integer index or by a mask"""
        out = OpDispatch.execute(OP.SELECT, self, slice_=slice_)
        return out

    def __add__(self, other):
        """Add tensor to other tensor or constant element-wise"""
        other = self._wrap_constant_maybe(other)
        out = OpDispatch.execute(OP.ADD, self, other)
        return out

    __radd__ = __add__

    def __sub__(self, other):
        """Subtract other tensor or constant from tensor element-wise"""
        other = self._wrap_constant_maybe(other)
        out = OpDispatch.execute(OP.SUB, self, other)
        return out

    def __rsub__(self, other):
        other = self._wrap_constant_maybe(other)
        return other - self
    
    def __mul__(self, other):
        """Multiply tensor with other tensor or constant element-wise"""
        other = self._wrap_constant_maybe(other)
        out = OpDispatch.execute(OP.MUL, self, other)
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        """Divide tensor by other tensor or constant element-wise"""
        other = self._wrap_constant_maybe(other)
        out = OpDispatch.execute(OP.DIV, self, other)
        return out

    def __rtruediv__(self, other):
        other = self._wrap_constant_maybe(other)
        return other / self

    def __pow__(self, value):
        """Raise tensor to the power"""
        assert isinstance(value, (int, float))
        out = OpDispatch.execute(OP.POW, self, value=value)
        return out
    
    def __neg__(self):
        """Negate tensor (multiply by -1.0)"""
        neg = self._wrap_constant_maybe(-1.0)
        out = OpDispatch.execute(OP.MUL, self, neg)
        return out

    def __invert__(self):
        """Invert boolean tensor"""
        out = OpDispatch.execute(OP.INVERT, self)
        return out

    def sqrt(self):
        """Compute square root element-wise"""
        out = self ** 0.5
        return out

    def exp(self):
        """Compute exponent element-wise"""
        out = OpDispatch.execute(OP.EXP, self)
        return out

    def log(self):
        """Compute natural logarithm element-wise"""
        out = OpDispatch.execute(OP.LOG, self)
        return out

    def sum(self, dim=None, keepdim=False):
        """Reduce tensor by computing sum over dimension or globally. Supports reduce over multiple dimensions."""
        out = OpDispatch.execute(OP.SUM_REDUCE, self, dim=dim, keepdim=keepdim)
        return out
    
    def mean(self, dim=None, keepdim=False):
        """Reduce tensor by computing mean over dimension or globally. Supports reduce over multiple dimensions."""
        out = OpDispatch.execute(OP.MEAN_REDUCE, self, dim=dim, keepdim=keepdim)
        return out
    
    def max(self, dim=None):
        """Reduce tensor by computing max over dimension or globally"""
        out = OpDispatch.execute(OP.MAX_REDUCE, self, dim=dim)
        return out

    def min(self, dim=None):
        """Reduce tensor by computing min over dimension or globally"""
        out = OpDispatch.execute(OP.MIN_REDUCE, self, dim=dim)
        return out

    def relu(self):
        """Compute ReLU function element-wise"""
        out = OpDispatch.execute(OP.RELU, self)
        return out
    
    def sigmoid(self):
        """Compute sigmoid function element-wise"""
        out = OpDispatch.execute(OP.SIGMOID, self)
        return out
    
    def softmax(self, dim):
        """Compute softmax function over 1D slice along given dimension"""
        out = OpDispatch.execute(OP.SOFTMAX, self, dim=dim)
        return out

    def reshape(self, *shape):
        """Reshape tensor to a new shape"""
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        out = OpDispatch.execute(OP.RESHAPE, self, shape=shape)
        return out
    
    def permute(self, *dims):
        """Permute tensor dimensions"""
        dims = dims[0] if len(dims) == 1 and isinstance(dims[0], (tuple, list)) else dims
        out = OpDispatch.execute(OP.PERMUTE, self, dims=dims)
        return out
    
    def transpose(self, *dims):
        """Permute tensor dimensions"""
        out = self.permute(*dims)
        return out

    def squeeze(self, dim):
        """Drop tensor dimension of size one"""
        out = OpDispatch.execute(OP.SQUEEZE, self, dim=dim)
        return out

    def unsqueeze(self, dim):
        """Add new dimension of size one"""
        out = OpDispatch.execute(OP.UNSQUEEZE, self, dim=dim)
        return out
    
    def concat(self, tensors, dim):
        """Concate tensor with other tensors along dimension"""
        tensors = [self, *tensors]
        out = OpDispatch.execute(OP.CONCAT, tensors, dim=dim)
        return out
    
    def cat(self, tensors, dim):
        """Concate tensor with other tensors along dimension"""
        out = self.concat(tensors, dim=dim)
        return out

    def masked_fill_(self, mask, value):
        """Fill tensor with value at positions where mask is `True`. Mask must be of the same shape as tensor."""
        out = OpDispatch.execute(OP.MASKED_FILL, self, mask=mask, value=value)
        return out
    
    def lookup(self, mask):
        """Lookup indices into tensor provided by mask"""
        out = OpDispatch.execute(OP.LOOKUP, self, mask=mask)
        return out

    def matmul(self, other):
        """Perform tensor multiplication with other tensor"""
        out = OpDispatch.execute(OP.MATMUL, self, other)
        return out
    
    def conv2d(self, kernel, bias=None, stride=None, padding=None):
        """Perform 2D convolution on tensor with given kernel and parameters"""
        children = (self, kernel, bias) if bias is not None else (self, kernel)
        out = OpDispatch.execute(OP.CONV2D, *children, stride=stride, padding=padding)
        return out
    
    def max_pool2d(self, kernel_size, stride=None, padding=None):
        """Perform 2D max pooling on tensor with given kernel and parameters"""
        children = (self,)
        out = OpDispatch.execute(OP.MAX_POOL2D, *children, kernel_size=kernel_size, stride=stride, padding=padding)
        return out
    
    def avg_pool2d(self, kernel_size, stride=None, padding=None):
        """Perform 2D average pooling on tensor with given kernel and parameters"""
        children = (self,)
        out = OpDispatch.execute(OP.AVG_POOL2D, *children, kernel_size=kernel_size, stride=stride, padding=padding)
        return out
    
    def copy(self):
        """Copy of the tensor"""
        ob = self._copy_from_data(self.data)
        return ob
    
    def detach(self):
        """Detach tensor from autograd tracking"""
        ob = self.copy()
        return ob

    def from_data(self, data):
        """Construct new tensor with given data"""
        ob = self._copy_from_data(data)
        return ob

    def zeros_like(self):
        """Construct new tensor of zeros with the same spec as tensor"""
        data = self._storage.zeros(self.data.shape, dtype=self.dtype)
        return self._copy_from_data(data)
    
    def ones_like(self):
        """Construct new tensor of ones with the same spec as tensor"""
        data = self._storage.ones(self.data.shape, dtype=self.dtype)
        return self._copy_from_data(data)
    
    def to(self, device, inplace=True):
        """Move tensor to device""" 
        if not inplace:
            ob = self._copy_partial(device=device)
            return ob
        data_numpy = self._storage.numpy(self.data)
        grad_numpy = self._storage.numpy(self.grad)
        storage = StorageDispatch.get(device)
        self.data = storage.tensor(data_numpy, dtype=self.dtype)
        self.grad = storage.tensor(grad_numpy, dtype=self.dtype)
        self._storage = storage
        return self
    
    def cpu(self, inplace=True):
        """Move tensor to CPU device""" 
        ob = self.to('cpu', inplace=inplace)
        return ob
    
    def cuda(self, inplace=True):
        """Move tensor to CUDA device""" 
        ob = self.to('cuda', inplace=inplace)
        return ob

    def float(self, inplace=False):
        """Cast tensor to float32""" 
        ob = self._cast(dtype=DTYPE.FLOAT32, inplace=inplace)
        return ob
    
    def bool(self, inplace=False):
        """Cast tensor to bool""" 
        ob = self._cast(dtype=DTYPE.BOOL, inplace=inplace)
        return ob
    
    def long(self, inplace=False):
        """Cast tensor to int64""" 
        ob = self._cast(dtype=DTYPE.INT64, inplace=inplace)
        return ob
    
    def numpy(self):
        """Convert tensor to numpy array preserving data type""" 
        data = self._storage.numpy(self.data)
        return data

    def tolist(self):
        """Convert tensor to native python list or scalar""" 
        return self.data.tolist()
    
    def item(self):
        """Convert tensor to native python list or scalar""" 
        return self.tolist()

    def render(self):
        """Render autograd graph from tensor as a leaf node"""
        from .util.render import render_graph
        image = render_graph(self)
        return image
    
    def backward(self, upstream=None, destroy_graph=True):
        """Run autograd backward pass starting from tensor as a leaf node"""
        if upstream is not None:
            upstream = self._storage.tensor(upstream, dtype=self.dtype)
        else:
            upstream = self._storage.ones(self.shape, dtype=self.dtype)
        self.grad = upstream
        nodes = self._traverse()
        for node in reversed(nodes):
            node._backward()
        if destroy_graph:
            for node in nodes:
                del node._op
                del node._children
                node._op = None
                node._children = ()

    def _cast(self, dtype, inplace):
        if inplace:
            self.data = self._storage.cast(self.data, dtype=dtype)
            rv = self
        else:
            rv = self._copy_partial(dtype=dtype)
        return rv

    def _copy_partial(self, data=None, dtype=None, device=None, requires_grad=None):
        ob = type(self)(
            data=data if data is not None else self.data.copy(),
            dtype=dtype or self.dtype,
            device=device or self.device,
            requires_grad=requires_grad or self.requires_grad,
        )
        return ob

    def _copy_from_data(self, data):
        ob = type(self)(
            data=data,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        return ob

    def _wrap_constant_maybe(self, value):
        if isinstance(value, (float, int)):
            value = self._storage.tensor(value, dtype=self.dtype)
            tensor = self._copy_from_data(value)
            tensor.requires_grad = False
            value = tensor
        return value
    
    def _traverse(self):
        nodes_sorted = []
        visited = set()
        
        def __traverse(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for ch in node._children:
                __traverse(ch)
            nodes_sorted.append(node)
        
        __traverse(self)
        return nodes_sorted
