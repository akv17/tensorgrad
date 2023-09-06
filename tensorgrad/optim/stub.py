from abc import ABC, abstractmethod


class Optimizer(ABC):

    @abstractmethod
    def zero_grad(self): pass

    @abstractmethod
    def step(self): pass
