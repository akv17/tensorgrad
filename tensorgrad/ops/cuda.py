from .np.generator import OpGenerator
from ..const import DEVICE
from ..util.lazy import LazyImport

_cupy = LazyImport('cupy', err_msg='Implementation of CUDA device is not available')
_generator = OpGenerator(runtime=_cupy, device=DEVICE.CUDA)
_generator.generate_ops()
