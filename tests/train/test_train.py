import os
import pickle
import unittest

import numpy as np

from tests.util import require_torch, check_tensors
import tensorgrad

torch = require_torch()
torch.manual_seed(0)


class TestTraining(unittest.TestCase):

    def test_mlp_classifier(self):
        with open(os.path.join('tests', 'train', 'data', 'clf.pkl'), 'rb') as f:
            _x, _y = pickle.load(f)
        _x = np.array(_x)
        _y = np.array(_y)
        
        num_features = _x.shape[-1]
        num_classes = len(set(_y))
        num_epochs = 10
        batch_size = 4
        lr = 0.01
        momentum = 0.9

        tx = torch.tensor(_x, requires_grad=False).float()
        ty = torch.tensor(_y, requires_grad=False).long()
        tmodel = torch.nn.Sequential(
            torch.nn.Linear(num_features, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4, num_classes),
        )
        tlayers = tmodel[::2]
        tloss = torch.nn.CrossEntropyLoss()
        toptim = torch.optim.SGD(tmodel.parameters(), lr=lr, momentum=momentum)

        x = tensorgrad.Tensor(_x, name='x', dtype=tensorgrad.DTYPE.FLOAT32, requires_grad=False)
        y = tensorgrad.Tensor(_y, name='y', dtype=tensorgrad.DTYPE.INT64, requires_grad=False)
        model = tensorgrad.nn.Sequential(
            tensorgrad.nn.Linear(num_features, 32).initialize(weight=tmodel[0].weight.data.tolist(), bias=tmodel[0].bias.data.tolist()),
            tensorgrad.nn.ReLU(),
            tensorgrad.nn.Linear(32, 16).initialize(weight=tmodel[2].weight.detach().numpy(), bias=tmodel[2].bias.detach().numpy()),
            tensorgrad.nn.Sigmoid(),
            tensorgrad.nn.Linear(16, 8).initialize(weight=tmodel[4].weight.detach().numpy(), bias=tmodel[4].bias.detach().numpy()),
            tensorgrad.nn.ReLU(),
            tensorgrad.nn.Linear(8, 4).initialize(weight=tmodel[6].weight.detach().numpy(), bias=tmodel[6].bias.detach().numpy()),
            tensorgrad.nn.Sigmoid(),
            tensorgrad.nn.Linear(4, num_classes).initialize(weight=tmodel[8].weight.detach().numpy(), bias=tmodel[8].bias.detach().numpy()),
        )
        layers = model[::2]
        loss = tensorgrad.nn.CrossEntropyLoss()
        optim = tensorgrad.nn.SGD(model.parameters(), lr=lr, momentum=momentum)

        print()
        step = 0
        for epoch in range(num_epochs):
            epoch += 1
            for i in range(0, len(_x), batch_size):
                step += 1
                
                txb = tx[i:i+batch_size]
                tyb = ty[i:i+batch_size]
                toptim.zero_grad()
                tout = tmodel(txb)
                tlossv = tloss(tout, tyb)
                tlossv.backward()
                toptim.step()

                xb = x[i:i+batch_size]
                yb = y[i:i+batch_size]
                optim.zero_grad()
                out = model(xb)
                lossv = loss(out, yb)
                lossv.backward()
                optim.step()

                print(f'epoch: {epoch} step: {step} torch_loss: {tlossv.item():.8f} tensorgrad_loss: {lossv.item():.8f}')
                self.assertTrue(check_tensors(tlossv, lossv), msg=f'forward@step{step}')

                for tlayer, layer in zip(tlayers, layers):
                    tlw = tlayer.weight.grad
                    tlb = tlayer.bias.grad
                    lw = layer.weight.grad
                    lb = layer.bias.grad
                    self.assertTrue(check_tensors(tlw, lw, tol=1e-5, show_diff=True), msg=f'w_grad@{layer.weight.shape}')
                    self.assertTrue(check_tensors(tlb, lb, tol=1e-5, show_diff=True), msg=f'b_grad@{layer.bias.shape}')
