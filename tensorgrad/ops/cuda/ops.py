from .cp import CupyProvider
from ..dispatch import OpDispatch
from ...const import OP, DEVICE
import cupy

class _OpBinder:

    def __init__(self):
        from .. import cpu
        self.op_map = {v.NAME: v for k, v in vars(cpu).items() if hasattr(v, 'NAME')}

    def bind(self, name):
        cpu_op = self.op_map[name]
        cuda_name = name.value.capitalize()
        methods = {'forward': cpu_op.forward, 'backward': cpu_op.backward}
        cuda_op = type(cuda_name, (cpu_op,), methods)
        cuda_op.np = cupy
        cuda_op = OpDispatch.register(name, DEVICE.CUDA)(cuda_op)
        return cuda_op


_binder = _OpBinder()

for name in OP:
    cuda_op = _binder.bind(name)
    globals()['cuda_op']

