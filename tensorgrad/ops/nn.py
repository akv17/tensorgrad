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
        # brief explanation of the implementation:
        # - softmax operates only on vectors.
        # - no matter how many dimensions input has, softmax output may always be seen as a collection of `k` softmax vectors.
        # - for inputs of `ndim>1` each resulting softmax vector is independent from any other.
        # - the same holds for the gradients.
        # - leveraging that we arrange computations around vectors.

        # 1. use softmax forward output tensor.
        # 2. swap `self.dim` dim with the last dim, so `self.dim` is now the last dim.
        # 3. flatten tensor to shape [-1, sizeOf(self.dim)] -> a collection of `k` vectors along `self.dim` dim.
        # 4. cast upstream grad tensor to the same shape (originally it has the same shape as softmax forward output).
        # 5. compute local grad as a collection of `k` jacobians -- separate jacobian for each of the `k` softmax vectors.
        #     for each of `k` vectors:
        #         5.1. number of inputs `n` is equal to the size of `self.dim`.
        #         5.2. number of outputs is equal to the number of inputs.
        #         5.3. thus jacobian has shape [n, n].
        #         5.4. element `i,j` is computed as: -softmax(j) * softmax(i)
        #         5.5. element `i,i` is computed as: softmax(i) * (1.0 - softmax(i))
        #         5.6  compute 5.4 as outer product.
        #         5.7. this settles down the jacobian.
        #         5.8  compute 5.5 and fill the jacobian diagonal with the result.
        # 6. now we have collection of `k` jacobians where `k` is the number of vectors along `self.dim`
        # 7. compute downstream grad by multiplying each jacobian with the upstream grad from step 4.
        # 8. cast downstream grad to the shape of softmax inputs (cast back to original shape from step 1)

        if not self.x.requires_grad:
            return
        
        import numpy as np
        # step: 1
        softmax = self.out.data.data
        shape = softmax.shape
        dim = self.dim
        dims = list(range(len(shape)))
        
        # step: 2
        transpose_dims = dims.copy()
        transpose_dims[dim] = dims[-1]
        transpose_dims[-1] = dim

        # step: 3
        softmax_t = np.transpose(softmax, transpose_dims)
        transpose_shape = softmax_t.shape
        softmax_t = softmax_t.reshape(-1, softmax_t.shape[-1])
        
        # step: 4
        u_grad_t = np.transpose(self.out.grad.data, transpose_dims)
        u_grad_t = u_grad_t.reshape(-1, u_grad_t.shape[-1])

        # step: 5
        # step: 5.6
        outer_a = np.expand_dims(softmax_t, -1)
        outer_b = np.expand_dims(softmax_t, 1)
        outer = np.matmul(-outer_a, outer_b)

        # step: 5.8
        d_ii = softmax_t * (1.0 - softmax_t)
        outer[:, range(shape[dim]), range(shape[dim])] = d_ii

        # step: 7
        d_grad = np.matmul(outer, np.expand_dims(u_grad_t, -1))
        
        # step:8 
        d_grad = d_grad.reshape(transpose_shape)
        d_grad = np.transpose(d_grad, transpose_dims)

        self.x.grad.data += d_grad



        # import numpy as np
        # softmax = self.out.data.data
        # shape = softmax.shape
        # dims = list(range(len(shape)))
        # dim = self.dim
        # t = dims.copy()
        # t[dim] = dims[-1]
        # t[-1] = dim
        # softmax_t = np.transpose(softmax, t)
        # t_shape = softmax_t.shape
        # softmax_t = softmax_t.reshape(-1, softmax_t.shape[-1])
        # grad_t = np.transpose(self.out.grad.data, t)
        # grad_t = grad_t.reshape(-1, grad_t.shape[-1])
        # lgrads = []
        # for i in range(len(softmax_t)):
        #     si = softmax_t[i]
        #     gi = grad_t[i]
        #     jacob = -(si.reshape(-1, 1)).dot(si.reshape(1, -1))
        #     d_ii = si * (1.0 - si)
        #     np.fill_diagonal(jacob, d_ii)
        #     gi = jacob.dot(gi)
        #     lgrads.append(gi)
        # lgrads = np.array(lgrads)
        # lgrads = lgrads.reshape(t_shape)
        # lgrads = np.transpose(lgrads, t)
        # self.x.grad.data += lgrads
        



        
        
        
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
