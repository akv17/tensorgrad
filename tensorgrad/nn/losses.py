class CrossEntropyLoss:

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def forward(self, outputs, targets):
        softmax = outputs.softmax(dim=-1)
        true_ixs = targets.tolist()
        pred = softmax[range(softmax.shape[0]), true_ixs]
        pred = pred.log()
        reduced = getattr(pred, self.reduction)()
        loss = -reduced
        return loss
