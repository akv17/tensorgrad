from .stub import Optimizer
from ..ctx import no_grad


class SGD(Optimizer):
    
    def __init__(self, parameters, lr, momentum=None):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum or 0.0
        
        self._step = 0
        self._momentum_buffer = [0.0] * len(self.parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = p.zeros_like().data

    def step(self):
        with no_grad():
            self._step += 1
            for pi, p in enumerate(self.parameters):
                g = p.from_data(p.grad)
                if self.momentum:
                    if self._step > 1:
                        b = self.momentum * self._momentum_buffer[pi] + g
                    else:
                        b = g
                    self._momentum_buffer[pi] = b
                    g = b
                u = -self.lr * g
                p.data += u.data
