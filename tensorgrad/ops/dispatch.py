from .unary import Pow, Exp, Log
from .binary import Add, Mul, Sub, Div
from .reduce import SumReduce, MeanReduce
from .nn import Relu, Sigmoid

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

    Relu,
    Sigmoid
)
_DISPATCH = {op.NAME: op for op in _OPS}


class OpDispatch:

    @classmethod
    def execute(cls, op, *args, **kwargs):
        op = _DISPATCH[op](*args, **kwargs)
        out = op.forward()
        out._children = args
        out._op = op
        return out
