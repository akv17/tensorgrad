from abc import ABC, abstractmethod


class Optimizer(ABC):
    
    @abstractmethod
    def step(self): pass
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = p.zeros_like().data
