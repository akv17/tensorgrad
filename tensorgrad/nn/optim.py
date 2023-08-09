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
