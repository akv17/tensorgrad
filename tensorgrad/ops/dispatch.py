from .unary import Pow, Exp, Log
from .binary import Add, Mul, Sub, Div
from .reduce import SumReduce, MeanReduce
from .shape import Unsqueeze, Squeeze, Reshape, Permute, Select
from .nn import Relu, Sigmoid, Softmax, Matmul

_OPS = (
    Pow,
    Exp,
    Log,
    
    Add,
    Mul,
    Sub,
    Div,
    
    SumReduce,
    MeanReduce,

    Squeeze,
    Unsqueeze,
    Reshape,
    Permute,
    Select,

    Relu,
    Sigmoid,
    Softmax,
    Matmul,
)
_DISPATCH = {op.NAME: op for op in _OPS}


class OpDispatch:

    @classmethod
    def execute(cls, op, *args, **kwargs):
        op = _DISPATCH[op](*args, **kwargs)
        out = op.forward()
        out.requires_grad = any(a.requires_grad for a in args)
        out._children = args
        out._op = op
        return out
