from .base import Optimizer
from ..ctx import no_grad


class Adam(Optimizer):
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.beta1, self.beta2 = self.betas
        self.eps = eps
        
        self._step = 0
        self._moment1_buffer = [None] * len(self.parameters)
        self._moment2_buffer = [None] * len(self.parameters)

    def step(self):
        with no_grad():
            self._step += 1
            for pi, p in enumerate(self.parameters):
                g = p.from_data(p.grad)
                m_prev = p.zeros_like() if self._step == 1 else self._moment1_buffer[pi]
                m = self.beta1 * m_prev + (1.0 - self.beta1) * g
                self._moment1_buffer[pi] = m
                v_prev = p.zeros_like() if self._step == 1 else self._moment2_buffer[pi]
                v = self.beta2 * v_prev + (1.0 - self.beta2) * (g ** 2)
                self._moment2_buffer[pi] = v
                m = m / (1.0 - self.beta1 ** (self._step))
                v = v / (1.0 - self.beta2 ** (self._step))
                u = -(self.lr * m) / (v.sqrt() + self.eps)
                p.data += u.data
