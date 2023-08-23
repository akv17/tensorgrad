from ..stubs import BinaryOp
from ..dispatch import OpDispatch
from ..util import accumulate_broadcasted_grad
from ...const import OP, DEVICE


@OpDispatch.register(op=OP.ADD, device=DEVICE.CPU)
class Add(BinaryOp):

    def forward(self):
        o = self.a.data + self.b.data
        self.out = self.a.from_data(o)
        return self.out
    
    def backward(self):
        if self.a.requires_grad:
            a_grad = self.out.grad
            a_grad = accumulate_broadcasted_grad(self.a, a_grad)
            self.a.grad += a_grad
        if self.b.requires_grad:
            b_grad = self.out.grad
            b_grad = accumulate_broadcasted_grad(self.b, b_grad)
            self.b.grad += b_grad


@OpDispatch.register(op=OP.MUL, device=DEVICE.CPU)
class Mul(BinaryOp):

    def forward(self):
        data = self.a.data * self.b.data
        self.out = self.a.from_data(data)
        return self.out

    def backward(self):
        if self.a.requires_grad:
            a_grad = self.b.data * self.out.grad
            a_grad = accumulate_broadcasted_grad(self.a, a_grad)
            self.a.grad += a_grad
        if self.b.requires_grad:
            b_grad = self.a.data * self.out.grad
            b_grad = accumulate_broadcasted_grad(self.b, b_grad)
            self.b.grad += b_grad
