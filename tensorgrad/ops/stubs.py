from abc import ABC, abstractmethod


class BaseOp(ABC):

    @abstractmethod
    def forward(self): pass

    @abstractmethod
    def backward(self): pass


class BinaryOp(BaseOp, ABC):

    def __init__(self, a, b):
        self.out = None
        self.a = a
        self.b = b

    @abstractmethod
    def forward(self): pass

    @abstractmethod
    def backward(self): pass


class ReduceOp(BaseOp, ABC):

    def __init__(self, x, *, dim=None):
        self.x = x
        self.dim = dim

    @abstractmethod
    def forward(self): pass

    @abstractmethod
    def backward(self): pass
