from .np.generator import OpGenerator, RuntimeProvider
from ..const import DEVICE

_runtime = RuntimeProvider(impl='cupy', device=DEVICE.CUDA)
_generator = OpGenerator(runtime=_runtime, device=DEVICE.CUDA)
_generator.generate_ops()
