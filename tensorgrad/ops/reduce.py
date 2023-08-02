from ..const import OP


class SumReduce:
    NAME = OP.SUM_REDUCE

    def __init__(self, out, a, dim=None):
        self.out = out
        self.a = a
        self.dim = dim

    def forward(self):
        self.out.data = self.a.data.sum(dim=self.dim)

    def backward(self):
        if self.a.requires_grad:
            self.a.grad += self.out.grad


class MeanReduce:
    NAME = OP.MEAN_REDUCE

    def __init__(self, out, a, dim=None):
        self.out = out
        self.a = a
        self.dim = dim

    def forward(self):
        sum_ = self.a.data.sum(dim=self.dim)
        size_ = self.a.shape[self.dim]
        self.out.data = sum_ / size_

    def backward(self):
        if self.a.requires_grad:
            self.a.grad += self.out.grad
