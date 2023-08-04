from .interface import Op
from ..const import OP


class Relu(Op):
    NAME = OP.RELU

    def __init__(self, x):
        self.out = None
        self.x = x
        self.mask = None

    def forward(self):
        self.out = self.x.copy()
        self.mask = self.x.data <= 0.0
        self.out.data = self.out.data.fill(self.mask, 0.0)
        return self.out

    def backward(self):
        if self.x.requires_grad:
            mask = self.x.ones_like()
            mask.data = mask.data.fill(self.mask, 0.0)
            self.x.grad += mask.data * self.out.grad


class Sigmoid(Op):
    NAME = OP.SIGMOID

    def __init__(self, x):
        self.out = None
        self.x = x

    def forward(self):
        self.out = self.x.zeros_like()
        self.out.data = 1.0 / (1.0 + (-self.x.data).exp())
        return self.out

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.out.data * (1.0 - self.out.data) * self.out.grad


class Softmax(Op):
    NAME = OP.SOFTMAX

    def __init__(self, x, *, dim):
        self.out = None
        self.x = x
        self.dim = dim

    def forward(self):
        exp = self.x.data.exp()
        norm = exp.sum(self.dim)
        norm = norm.unsqueeze(self.dim)
        self.exp = exp
        self.norm = norm
        out_data = exp / norm
        self.out = self.x.zeros(out_data.shape)
        self.out.data = out_data
        self.out.grad = out_data.zeros_like()
        return self.out

    def backward(self):
        # here we have to compute full jacobian as each output depends on each input.
        # df/di when i != j: -softmax(j) * softmax(i)
        # df/di grad when i == j: softmax(i) * (1.0 - softmax(i))

        if not self.x.requires_grad:
            return

        import numpy as np
        softmax = self.out.data.data
        shape = softmax.shape
        dims = list(range(len(shape)))
        dim = self.dim
        t = dims.copy()
        t[dim] = dims[-1]
        t[-1] = dim
        softmax_t = np.transpose(softmax, t)
        t_shape = softmax_t.shape
        softmax_t = softmax_t.reshape(-1, softmax_t.shape[-1])
        grad_t = np.transpose(self.out.grad.data, t)
        grad_t = grad_t.reshape(-1, grad_t.shape[-1])
        lgrads = []
        for i in range(len(softmax_t)):
            si = softmax_t[i]
            gi = grad_t[i]
            jacob = -(si.reshape(-1, 1)).dot(si.reshape(1, -1))
            d_ii = si * (1.0 - si)
            np.fill_diagonal(jacob, d_ii)
            gi = jacob.dot(gi)
            lgrads.append(gi)
        lgrads = np.array(lgrads)
        lgrads = lgrads.reshape(t_shape)
        lgrads = np.transpose(lgrads, t)
        self.x.grad.data += lgrads
        



        
        
        
        # if self.x.requires_grad:
        #     softmax = self.out.data
        #     ndim = softmax.data.ndim
        #     if ndim == 1:
        #         jacob = -(softmax.unsqueeze(1)).matmul(softmax.unsqueeze(0))
        #         d_ii = softmax * (1.0 - softmax)
        #         jacob = jacob.fill_diagonal(d_ii)
        #         grad = jacob.matmul(self.out.grad)
        #         self.x.grad += grad
        #     else:
        #         if self.dim == ndim - 1:
        #             jacob = -softmax.matmul(softmax)
        #         else:
        #             shape = softmax.shape
        #             transpose = shape
        #             transpose[self.dim] = shape[self.dim + 1]
        #             transpose[self.dim + 1] = shape[self.dim]
        #             softmax = softmax.permute(transpose)
        #             jacob = -softmax.matmul(softmax)

                    # tmp = sof
