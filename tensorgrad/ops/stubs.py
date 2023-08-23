from abc import ABC, abstractmethod


class BinaryOp(ABC):

    def __init__(self, a, b):
        self.out = None
        self.a = a
        self.b = b

    @abstractmethod
    def forward(self): pass

    @abstractmethod
    def backward(self): pass
