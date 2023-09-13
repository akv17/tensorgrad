import unittest

from .util import check_tensors, require_torch
from .const import DEVICE, SHOW_DIFF

torch = require_torch()


class CommonHelper(unittest.TestCase):

    def _check_tensors(self, pairs):
        for tt, t, tol, name in pairs:
            self.assertTrue(check_tensors(tt.tolist(), t.tolist(), tol=tol, show_diff=SHOW_DIFF), msg=name)
    
    def _backward(self, tensor):
        # before calling backward we multiply each scalar with unique number
        # to make sure that each scalar in the grad is indeed computed correctly.
        # for example just calling sum() will hide incorrect broadcasting because upstream is all ones.
        # on the contrary each scalar in arange'd upstream is unique and such an error will be easily revealed.
        if isinstance(tensor, torch.Tensor):
            range_ = torch.arange(tensor.numel())
        else:
            range_ = tensor.arange(tensor.numel(), device=DEVICE)
        range_ = range_.float()
        range_ = range_.reshape(tensor.shape) + 1.0
        norm = range_.data.max().tolist()
        range_ = range_ / norm
        o = (tensor * range_).sum()
        o.backward()
