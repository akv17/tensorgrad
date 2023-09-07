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

    def _check_input(self, outputs, targets):
        if outputs.ndim != 2:
            msg = f'expected outputs to have 2 dimensions but got {outputs.ndim}.'
            raise Exception(msg)
        if targets.ndim != 1:
            msg = f'expected targets to have 1 dimension but got {targets.ndim}.'
            raise Exception(msg)
        if outputs.shape[0] != targets.shape[0]:
            msg = f'expected outputs and targets to have same number of samples but got {(outputs.shape[0], targets.shape[0])}.'
            raise Exception(msg)
        if outputs.shape[0] != targets.shape[0]:
            msg = f'expected outputs and targets to have same number of samples but got {(outputs.shape[0], targets.shape[0])}.'
            raise Exception(msg)



class MSELoss(Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs, targets):
        loss = ((outputs - targets) ** 2).mean()
        return loss
    
    def init_from_torch(self, module):
        pass
