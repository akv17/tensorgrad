from ..const import OP


class SumReduce:
    NAME = OP.SUM_REDUCE

    def __init__(self, out, a, dim=0):
        self.out = out
        self.a = a
        self.dim = dim

    def forward(self):
        self.out.data = self.a.data.sum(dim=self.dim)

    def backward(self):
        if self.a.requires_grad:
            self.a.grad += self.out.grad
