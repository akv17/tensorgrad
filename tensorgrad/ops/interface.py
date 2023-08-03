from abc import ABC, abstractmethod

class Op(ABC):
    
    @property
    @abstractmethod
    def NAME(self): pass

    @abstractmethod
    def forward(self): pass
    
    @abstractmethod
    def backward(self): pass
