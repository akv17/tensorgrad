import math
from .base import Module, Parameter
from .. import init


class CrossEntropyLoss:

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        return out

    def forward(self, outputs, targets):
        softmax = outputs.softmax(dim=-1)
        true_ixs = targets.tolist()
        pred = softmax[range(softmax.shape[0]), true_ixs]
        pred = pred.log()
        reduced = getattr(pred, self.reduction)()
        loss = -reduced
        return loss


class MSELoss:

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        return out

    def forward(self, outputs, targets):
        loss = ((outputs - targets) ** 2).mean()
        return loss