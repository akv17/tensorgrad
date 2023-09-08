from .np.generator import OpGenerator, RuntimeProvider
from ..const import DEVICE

_runtime = RuntimeProvider(impl='numpy', device=DEVICE.CPU)
_generator = OpGenerator(runtime=_runtime, device=DEVICE.CPU)
_generator.generate_ops()
