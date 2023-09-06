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
