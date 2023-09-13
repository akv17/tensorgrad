import functools
import operator

from .base import Module, Parameter
from .. import init


class Embedding(Module):
    """
    Performs embedding table lookup.  
    Expects inputs to contain indices into the embedding table.  

    **Parameters:**
    - `num_embeddings: int:` number of embeddings (aka number of rows in the table)
    - `embedding_dim: int:` dimensionality of each embedding (aka number of columns in the table)

    **Input:** `(*)`  
    **Output:** `(*, E)` attaches new dimension with embeddings  
    **Weights:**  
    - `weight: (num_embeddings, embedding_dim):` learnable embedding table
    """

    def __init__(self, num_embeddings, embedding_dim, dtype=None, device=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter.empty((self.num_embeddings, self.embedding_dim), dtype=dtype, device=device)
        self.reset_parameters()
    
    def forward(self, x):
        o = self.weight.lookup(x)
        return o
    
    def reset_parameters(self):
        init.uniform_fan_in(self.weight)
    
    def init_from_torch(self, module):
        self.weight = Parameter(
            module.weight.detach().cpu().numpy(),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        return self


class Dropout(Module):
    """
    Applies dropout to input.  
    Basically, zeroes out each scalar element of the input with probability `p`.  
    Operates only in training mode.  

    **Parameters:**  
    - `p: int:` probability of element zeroing  

    **Input:** `(*)`  
    **Output:** `(*)`  
    **Weights:** `None`  
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.scale = 1.0 / (1.0 - self.p)
    
    def forward(self, x):
        if self.training:
            mask = self._generate_mask(x)
            x *= mask * self.scale
        return x
    
    def _generate_mask(self, x):
        # we invert mask and cast it to float to perform masking via elementwise multiplication.
        # positions with True in the original bool mask are meant to be zeroed out.
        mask = (~(x.bernoulli(p=self.p, shape=x.shape).bool())).float()
        mask.to(x.device)
        return mask


class Flatten(Module):
    """
    Flattens batch of input tensors to a 2D tensor preserving batch size.  
    Basically, flattens all dimensions except the first one.  

    **Parameters:** `None`  
    **Input:** `(B, *)`  
    **Output:** `(B, F)`  
    **Weights:** `None`  
    """
    
    def forward(self, x):
        batch_size = x.shape[0]
        num_features = functools.reduce(operator.mul, x.shape[1:], 1)
        x = x.reshape(batch_size, num_features)
        return x
