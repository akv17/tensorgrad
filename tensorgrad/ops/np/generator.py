"""
utilities to auto-generate numpy-based ops with specific numpy-compatible runtime.
currently NumPy runtime powers CPU ops while CuPy runtime powers CUDA ops.
"""

from .activation import *
from .binary import *
from .conv2d import *
from .index import *
from .matmul import *
from .pool import *
from .reduce import *
from .shape import *
from .softmax import *
from .unary import *

from ..dispatch import OpDispatch


class OpGenerator:

    def __init__(self, runtime, device):
        self.runtime = runtime
        self.device = device
        self._op_map = {v._NAME: v for v in globals().values() if hasattr(v, '_NAME')}
    
    def generate_ops(self):
        for op_name in self._op_map:
            op = self._generate_op(op_name)
            self._register_op(op=op, name=op_name)

    def _generate_op(self, name):
        impl = self._op_map[name]
        name = impl._NAME.value.capitalize()
        methods = {'forward': impl.forward, 'backward': impl.backward}
        op = type(name, (impl,), methods)
        op.np = self.runtime
        return op

    def _register_op(self, op, name):
        OpDispatch.register(name, self.device)(op)
