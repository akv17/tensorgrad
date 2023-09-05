from .util.np import NumpyNamespaceProvider
from ..stubs import BaseOp
from ..dispatch import OpDispatch
from ...const import OP, DEVICE


@OpDispatch.register(OP.SOFTMAX, DEVICE.CPU)
class Softmax(BaseOp, NumpyNamespaceProvider):

    def __init__(self, x, *, dim):
        self.out = None
        self.x = x
        self.dim = dim

    def forward(self):
        exp = self.np.exp(self.x.data)
        norm = exp.sum(self.dim)
        norm = self.np.expand_dims(norm, self.dim)
        out_data = exp / norm
        self.out = self.x.from_data(out_data)
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
        softmax_t = self.np.transpose(softmax, transpose_dims)
        transpose_shape = softmax_t.shape
        softmax_t = softmax_t.reshape(-1, softmax_t.shape[-1])
        
        # step: 4
        u_grad = self.out.grad
        u_grad_t = self.np.transpose(u_grad, transpose_dims)
        u_grad_t = u_grad_t.reshape(-1, u_grad_t.shape[-1])

        # step: 5
        # step: 5.6
        outer_a = self.np.expand_dims(softmax_t, -1)
        outer_b = self.np.expand_dims(softmax_t, 1)
        outer = -self.np.matmul(outer_a, outer_b)
        jacob = outer

        # step: 5.8
        jacob_ii = softmax_t * (1.0 - softmax_t)
        jacob[..., range(jacob.shape[-2]), range(jacob.shape[-1])] = jacob_ii

        # step: 7
        u_grad_t = self.np.expand_dims(u_grad_t, -1)
        d_grad = self.np.matmul(jacob, u_grad_t)
        
        # step:8 
        d_grad = d_grad.reshape(transpose_shape)
        d_grad = self.np.transpose(d_grad, transpose_dims)

        self.x.grad += d_grad
