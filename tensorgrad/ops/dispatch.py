from .unary import Pow, Exp, Log
from .binary import Add, Mul, Sub, Div
from .reduce import SumReduce, MeanReduce

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
)
_DISPATCH = {op.NAME: op for op in _OPS}


class OpDispatch:

    @classmethod
    def execute(cls, op, *args, **kwargs):
        out = args[0].zeros_like()
        op = _DISPATCH[op](out, *args, **kwargs)
        op.forward()
        out._children = args
        out._op = op
        return out
