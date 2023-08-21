from .interface import Op
from .util import accumulate_broadcasted_grad
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
        out_data = exp / norm
        self.out = self.x.zeros_like()
        self.out.data = out_data
        return self.out

    def backward(self):
        # several notes on the implementation:
        #
        # - softmax operates only on vectors.
        # - no matter how many dimensions input has, softmax output may always be seen as a collection of `k` softmax vectors.
        # - for inputs of `ndim > 1` each resulting softmax vector is independent from any other.
        # - the same holds for the gradients.
        # - leveraging that we arrange computations around vectors.
        #
        # 1. use softmax forward output tensor.
        # 2. swap `self.dim` dim with the last dim, so `self.dim` is now the last dim.
        # 3. flatten tensor to 2d shape [-1, sizeOf(self.dim)] -> a collection of `k` vectors along `self.dim` dim.
        # 4. cast upstream grad tensor to the same shape (originally it has the same shape as softmax forward output).
        # 5. compute local grad as a collection of `k` jacobians -- a separate jacobian for each of the `k` softmax vectors.
        #     for each of `k` vectors:
        #         5.1. number of inputs `n` is equal to the size of `self.dim`.
        #         5.2. number of outputs is equal to the number of inputs.
        #         5.3. thus jacobian has shape [n, n].
        #         5.4. element `i,j` is computed as: -softmax(j) * softmax(i)
        #         5.5. element `i,i` is computed as: softmax(i) * (1.0 - softmax(i))
        #         5.6  compute 5.4 as outer product.
        #         5.7. this populates the jacobian.
        #         5.8  compute 5.5 and fill the jacobian diagonal with it.
        # 6. now we have the collection of `k` jacobians where `k` is the number of vectors along `self.dim`
        #    6.1. shape of this jacobian is [k, n, n], where `n` is `sizeOf(self.dim)` and `k` is defined above.
        #    6.2. `jacobian[i, j, :]` is the local gradient (vector of size `n`) for the j-th input of i-th softmax vector.
        # 7. compute downstream grad by multiplying each jacobian with the broadcasted upstream grad from step 4.
        #    7.1. `jacobian[i, j, :].dot(upstream[i])` is the downstream gradient scalar value for the j-th input of i-th softmax vector.
        # 8. cast downstream grad to the shape of softmax inputs (cast back to original shape from step 1)

        if not self.x.requires_grad:
            return
        
        # step: 1
        softmax = self.out.data
        shape = softmax.shape
        dim = self.dim
        dims = list(range(len(shape)))
        
        # step: 2
        transpose_dims = dims.copy()
        transpose_dims[dim] = dims[-1]
        transpose_dims[-1] = dim

        # step: 3
        softmax_t = softmax.permute(transpose_dims)
        transpose_shape = softmax_t.shape
        softmax_t = softmax_t.reshape(-1, softmax_t.shape[-1])
        
        # step: 4
        u_grad = self.out.grad
        u_grad_t = u_grad.permute(transpose_dims)
        u_grad_t = u_grad_t.reshape(-1, u_grad_t.shape[-1])

        # step: 5
        # step: 5.6
        outer_a = softmax_t.unsqueeze(-1)
        outer_b = softmax_t.unsqueeze(1)
        outer = -outer_a.matmul(outer_b)
        jacob = outer

        # step: 5.8
        jacob_ii = softmax_t * (1.0 - softmax_t)
        jacob = jacob.fill_diagonal2d(jacob_ii)

        # step: 7
        u_grad_t = u_grad_t.unsqueeze(-1)
        d_grad = jacob.matmul(u_grad_t)
        
        # step:8 
        d_grad = d_grad.reshape(transpose_shape)
        d_grad = d_grad.permute(transpose_dims)

        self.x.grad += d_grad


class Matmul:
    NAME = OP.MATMUL

    def __init__(self, a, b):
        self.out = None
        self.a = a
        self.b = b
    
    def forward(self):
        c = self.a.data.matmul(self.b.data)
        self.out = self.a.from_data(c)
        return self.out

    def backward(self):
        # notes on handling multidim tensors:
        # - no matter how many dims tensors have, this op may always be seen as collection of 2d matmuls.
        # - to leverage this we reshape all tensors to 3d shape [-1, prelast_dim, last_dim]
        # - then computing local grads is simple:
        #       downstream_grad_a = upstream_grad * b.T
        #       downstream_grad_b = a.T * upstream_grad
        # - finally we reshape all downstream grads back to match shape of the inputs.

        if self.a.requires_grad:
            u_grad = self.out.grad
            u_grad = u_grad.reshape(-1, u_grad.shape[-2], u_grad.shape[-1])
            b = self.b.data
            b = b.reshape(-1, b.shape[-2], b.shape[-1])
            b = b.permute([0, 2, 1])
            d_grad = u_grad.matmul(b)
            d_grad = accumulate_broadcasted_grad(self.a, d_grad)
            d_grad = d_grad.reshape(self.a.shape)
            self.a.grad += d_grad
        if self.b.requires_grad:
            u_grad = self.out.grad
            u_grad = u_grad.reshape(-1, u_grad.shape[-2], u_grad.shape[-1])
            a = self.a.data
            a = a.reshape(-1, a.shape[-2], a.shape[-1])
            a = a.permute([0, 2, 1])
            d_grad = a.matmul(u_grad)
            d_grad = accumulate_broadcasted_grad(self.b, d_grad)
            d_grad = d_grad.reshape(self.b.shape)
            self.b.grad += d_grad



class Conv2D(Op):
    NAME = OP.CONV2D

    def __init__(self, x, kernel, bias=None, *, stride=None, padding=None):
        self.out = None
        self.x = x
        self.kernel = kernel
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def forward(self):
        bias = self.bias.data if self.bias is not None else None
        data = self.x.data.conv2d(self.kernel.data, bias=bias, stride=self.stride, padding=self.padding)
        self.out = self.x.from_data(data)
        return self.out

    def backward(self):
        pass
