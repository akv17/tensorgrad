class SGD:
    
    def __init__(self, parameters, lr, momentum=None, name=None):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum or 0.0
        self.name = name or f'sgd@{id(self)}'
        self._step = 0
        self._momentum_buffer = [0.0] * len(self.parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = p.grad.zeros_like()

    def step(self):
        self._step += 1
        for pi, p in enumerate(self.parameters):
            g = p.grad
            if self.momentum:
                if self._step > 1:
                    b = self.momentum * self._momentum_buffer[pi] + g
                else:
                    b = g
                self._momentum_buffer[pi] = b
                g = b
            p.data -= self.lr * g


class Adam:
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, name=None):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.beta1, self.beta2 = self.betas
        self.eps = eps
        self.name = name or f'adam@{id(self)}'
        self._step = 0
        self._moment1_buffer = [None] * len(self.parameters)
        self._moment2_buffer = [None] * len(self.parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = p.grad.zeros_like()

    def step(self):
        self._step += 1
        for pi, p in enumerate(self.parameters):
            g = p.grad
            m_prev = g.zeros_like() if self._step == 1 else self._moment1_buffer[pi]
            m = self.beta1 * m_prev + (1.0 - self.beta1) * g
            self._moment1_buffer[pi] = m
            v_prev = g.zeros_like() if self._step == 1 else self._moment2_buffer[pi]
            v = self.beta2 * v_prev + (1.0 - self.beta2) * (g ** 2)
            self._moment2_buffer[pi] = v
            m = m / (1.0 - self.beta1 ** (self._step))
            v = v / (1.0 - self.beta2 ** (self._step))
            upd = (self.lr * m) / (v.sqrt() + self.eps)
            p.data -= upd
