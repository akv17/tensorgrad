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
        self.out.data = self.a.data.mean(dim=self.dim)

    def backward(self):
        if self.a.requires_grad:
            size = self.a.data.size if self.dim is None else self.a.data.shape[self.dim]
            self.a.grad += 1.0 / size * self.out.grad
