from .tensor import Tensor
from . import nn
from . import optim
from .ctx import no_grad, is_grad_enabled, is_cuda_available
from .const import DEVICE, DTYPE

tensor = Tensor
