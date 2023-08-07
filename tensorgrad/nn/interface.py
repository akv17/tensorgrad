from abc import ABC, abstractmethod

class Module:

    def __call__(self):
        pass

    def forward(self, *args, **kwargs):
        pass

    def parameters(self):
        pass
