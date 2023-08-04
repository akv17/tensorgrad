from ..const import OP


class SumReduce:
    NAME = OP.SUM_REDUCE

    def __init__(self, x, dim=None):
        self.out = None
        self.x = x
        self.dim = dim

    def forward(self):
        self.out = self.x.zeros_like()
        self.out.data = self.x.data.sum(dim=self.dim)
        self.out.grad = self.out.data.zeros_like()
        return self.out

    def backward(self):
        if self.x.requires_grad:
            out_grad = self.out.grad.unsqueeze(self.dim) if self.dim is not None else self.out.grad
            self.x.grad += out_grad


class MeanReduce:
    NAME = OP.MEAN_REDUCE

    def __init__(self, x, dim=None):
        self.out = None
        self.x = x
        self.dim = dim

    def forward(self):
        self.out = self.x.zeros_like()
        self.out.data = self.x.data.mean(dim=self.dim)
        self.out.grad = self.out.data.zeros_like()
        return self.out

    def backward(self):
        if self.x.requires_grad:
            size = self.x.data.size if self.dim is None else self.x.data.shape[self.dim]
            out_grad = self.out.grad.unsqueeze(self.dim) if self.dim is not None else self.out.grad
            self.x.grad += 1.0 / size * out_grad