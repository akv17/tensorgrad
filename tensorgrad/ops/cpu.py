from .np.generator import OpGenerator
from ..const import DEVICE
from ..util.lazy import LazyImport

_numpy = LazyImport('numpy', err_msg='Implementation of CPU device is not available')
_generator = OpGenerator(runtime=_numpy, device=DEVICE.CPU)
_generator.generate_ops()
