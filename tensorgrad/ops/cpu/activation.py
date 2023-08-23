from .util import get_numpy
from ..stubs import UnaryOp
from ..dispatch import OpDispatch
from ...const import OP, DEVICE


@OpDispatch.register(OP.RELU, DEVICE.CPU)
class ReLU(UnaryOp):
    
    def forward(self):
        data = self.x.data.copy()
        self.mask = data <= 0.0
        data[self.mask] = 0.0
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            u = self.out.grad.copy()
            u[self.mask] = 0.0
            self.x.grad += u


@OpDispatch.register(OP.SIGMOID, DEVICE.CPU)
class Sigmoid(UnaryOp):

    def forward(self):
        np = get_numpy()
        data = 1.0 / (1.0 + np.exp(-self.x.data))
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.out.data * (1.0 - self.out.data) * self.out.grad
