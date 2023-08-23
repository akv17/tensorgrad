from ..stubs import BaseOp
from ..dispatch import OpDispatch
from ...const import OP, DEVICE


@OpDispatch.register(OP.RESHAPE, DEVICE.CPU)
class Reshape(BaseOp):

    def __init__(self, x, *, shape):
        self.out = None
        self.x = x
        self.shape = shape

    def forward(self):
        data = self.x.data.reshape(self.shape)
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.out.grad.reshape(self.x.grad.shape)
