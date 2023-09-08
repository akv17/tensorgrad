"""
cuda ops are generated automatically from cpu ops thanks to cupy / numpy interoperability.
though had to introduce couple of awkward solutions to actually achieve this.
cupy is not enforced to be installed until first cuda op call. 
"""

from .. import cpu
from ..dispatch import OpDispatch
from ...const import DEVICE, OP


class _CupyProvider:

    def __init__(self):
        self.__cupy = None

    def __getattr__(self, name):
        self.__import_maybe()
        return getattr(self.__cupy, name)
    
    def __import_maybe(self):
        if self.__cupy is None:
            try:
                import cupy
                self.__cupy = cupy
            except ImportError:
                msg = 'CUDA unavailable: cannot import CuPy'
                raise Exception(msg)


class _OpGenerator:

    def __init__(self):
        self.cpu_op_map = {v.NAME: v for v in vars(cpu).values() if hasattr(v, 'NAME')}
        self.cupy = _CupyProvider()

    def generate(self):
        for name in OP:
            op = self._generate_op(name)
            self._register_op(op)

    def _generate_op(self, name):
        cpu_op = self.cpu_op_map[name]
        cuda_name = name.value.capitalize()
        cuda_methods = {'forward': cpu_op.forward, 'backward': cpu_op.backward}
        cuda_op = type(cuda_name, (cpu_op,), cuda_methods)
        cuda_op.np = self.cupy
        cuda_op.NAME = name
        return cuda_op

    def _register_op(self, op):
        OpDispatch.register(op.NAME, DEVICE.CUDA)(op)


_OpGenerator().generate()
