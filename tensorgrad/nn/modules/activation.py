from .base import Module


class Identity(Module):

    def forward(self, x):
        return x
    
    def init_from_torch(self, module):
        pass


class ReLU(Module):
    
    def forward(self, x):
        x = x.relu()
        return x
    
    def init_from_torch(self, module):
        pass


class Sigmoid(Module):

    def forward(self, x):
        x = x.sigmoid()
        return x
    
    def init_from_torch(self, module):
        pass
