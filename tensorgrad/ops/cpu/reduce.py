from .util import get_numpy
from ..stubs import ReduceOp
from ..dispatch import OpDispatch
from ...const import OP, DEVICE


@OpDispatch.register(OP.SUM_REDUCE, DEVICE.CPU)
class SumReduce(ReduceOp):
    
    def forward(self):
        data = self.x.data.sum(self.dim)
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            np = get_numpy()
            out_grad = self.out.grad
            if self.dim is None:
                out_grad = self.out.grad
            elif self.dim == 0 and self.x.ndim < 2:
                out_grad = self.out.grad
            else:
                # out_grad = self.out.grad.unsqueeze(self.dim)
                out_grad = np.expand_dims(self.out.grad, self.dim)
            self.x.grad += out_grad
