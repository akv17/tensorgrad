from .base import Module


class Identity(Module):

    def forward(self, x):
        return x


class ReLU(Module):
    
    def forward(self, x):
        x = x.relu()
        return x


class Sigmoid(Module):

    def forward(self, x):
        x = x.sigmoid()
        return x


class Softmax(Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.softmax(dim=self.dim)
        return x
