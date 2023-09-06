import math
from .base import Module, Parameter
from .. import init


class CrossEntropyLoss(Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs, targets):
        softmax = outputs.softmax(dim=-1)
        true_ixs = targets.tolist()
        pred = softmax[range(softmax.shape[0]), true_ixs]
        pred = pred.log()
        reduced = getattr(pred, self.reduction)()
        loss = -reduced
        return loss
    
    def init_from_torch(self, module):
        pass


class MSELoss(Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs, targets):
        loss = ((outputs - targets) ** 2).mean()
        return loss
    
    def init_from_torch(self, module):
        pass
