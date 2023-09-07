import functools
import operator

from .base import Module, Parameter
from .. import init


class Embedding(Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, dtype=None, device=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
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

    def __init__(self, p=0.5):
        self.p = p
        self.scale = 1.0 / (1.0 - self.p)
    
    def forward(self, x):
        mask = self._generate_mask(x)
        x *= mask * self.scale
        return x
    
    def _generate_mask(self, x):
        # we invert mask and cast it to float to perform masking via elementwise multiplication.
        # positions with True in the original bool mask are meant to be zeroed out.
        mask = (~x.bernoulli(p=self.p, shape=x.shape)).float()
        return mask


class Flatten(Module):
    
    def forward(self, x):
        batch_size = x.shape[0]
        num_features = functools.reduce(operator.mul, x.shape[1:], 1)
        x = x.reshape(batch_size, num_features)
        return x
