import os
import pickle
import unittest

import numpy as np
from parameterized import parameterized

from tests.util import require_torch, check_tensors, generate_cases, get_device, get_dtype

import tensorgrad
torch = require_torch()
torch.manual_seed(0)

DEVICE = get_device()
DTYPE = get_dtype()
SHOW_DIFF = os.getenv('TESTS_SHOW_DIFF') == '1'
VERBOSE = os.getenv('TESTS_VERBOSE') == '1'


class TestTraining(unittest.TestCase):

    def setUp(self) -> None:
        self.helper = Helper()

    def test_mlp_classifier(self):
        with open(os.path.join('tests', 'data', 'clf.pkl'), 'rb') as f:
            _x, _y = pickle.load(f)
        _x = np.array(_x)
        _y = np.array(_y)
        
        num_features = _x.shape[-1]
        num_classes = len(set(_y))
        num_epochs = 10
        batch_size = 4
        lr = 0.01
        momentum = 0.9

        tx = torch.tensor(_x).float()
        ty = torch.tensor(_y).long()
        tm = torch.nn.Sequential(
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
        tloss_fn = torch.nn.CrossEntropyLoss()
        toptim = torch.optim.SGD(tm.parameters(), lr=lr, momentum=momentum)

        x = tensorgrad.Tensor(_x, dtype=DTYPE.FLOAT32, requires_grad=False)
        y = tensorgrad.Tensor(_y, dtype=DTYPE.INT32, requires_grad=False)
        m = tensorgrad.nn.Sequential(
            tensorgrad.nn.Linear(num_features, 32).init_from_torch(tm[0]),
            tensorgrad.nn.ReLU(),
            tensorgrad.nn.Linear(32, 16).init_from_torch(tm[2]),
            tensorgrad.nn.Sigmoid(),
            tensorgrad.nn.Linear(16, 8).init_from_torch(tm[4]),
            tensorgrad.nn.ReLU(),
            tensorgrad.nn.Linear(8, 4).init_from_torch(tm[6]),
            tensorgrad.nn.Sigmoid(),
            tensorgrad.nn.Linear(4, num_classes).init_from_torch(tm[8]),
        )
        loss_fn = tensorgrad.nn.CrossEntropyLoss()
        optim = tensorgrad.optim.SGD(m.parameters(), lr=lr, momentum=momentum)

        self.helper._train(
            tx, ty,
            tm, tloss_fn, toptim,
            x, y,
            m, loss_fn, optim,
            num_epochs,
            batch_size
        )
    
    def test_mlp_regressor(self):
        with open(os.path.join('tests', 'data', 'reg.pkl'), 'rb') as f:
            _x, _y = pickle.load(f)
        _x = np.array(_x)
        _y = np.array(_y)
        
        num_features = _x.shape[-1]
        num_epochs = 10
        batch_size = 4
        lr = 0.01
        momentum = 0.9

        tx = torch.tensor(_x).float()
        ty = torch.tensor(_y).float()
        tm = torch.nn.Sequential(
            torch.nn.Linear(num_features, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4, 1),
        )
        tloss_fn = torch.nn.MSELoss()
        toptim = torch.optim.SGD(tm.parameters(), lr=lr, momentum=momentum)

        x = tensorgrad.Tensor(_x, dtype=DTYPE.FLOAT32, requires_grad=False)
        y = tensorgrad.Tensor(_y, dtype=DTYPE.FLOAT32, requires_grad=False)
        m = tensorgrad.nn.Sequential(
            tensorgrad.nn.Linear(num_features, 32).init_from_torch(tm[0]),
            tensorgrad.nn.ReLU(),
            tensorgrad.nn.Linear(32, 16).init_from_torch(tm[2]),
            tensorgrad.nn.Sigmoid(),
            tensorgrad.nn.Linear(16, 8).init_from_torch(tm[4]),
            tensorgrad.nn.ReLU(),
            tensorgrad.nn.Linear(8, 4).init_from_torch(tm[6]),
            tensorgrad.nn.Sigmoid(),
            tensorgrad.nn.Linear(4, 1).init_from_torch(tm[8]),
        )
        loss_fn = tensorgrad.nn.MSELoss()
        optim = tensorgrad.optim.SGD(m.parameters(), lr=lr, momentum=momentum)

        self.helper._train(
            tx, ty,
            tm, tloss_fn, toptim,
            x, y,
            m, loss_fn, optim,
            num_epochs,
            batch_size
        )
    
    def test_cnn_classifier(self):
        with open(os.path.join('tests', 'data', 'mnist100.pkl'), 'rb') as f:
            _x, _y = pickle.load(f)
        _x = np.array(_x)
        _x = np.expand_dims(_x, 1)
        _y = np.array(_y)
        
        num_epochs = 10
        batch_size = 4

        tx = torch.tensor(_x).float()
        ty = torch.tensor(_y).long()
        tm = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=(3, 3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d((2, 2)),
            torch.nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=(3, 3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d((2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(8 * 7 * 7, 10),
        )
        tloss_fn = torch.nn.CrossEntropyLoss()
        toptim = torch.optim.SGD(tm.parameters(), 0.01)

        x = tensorgrad.Tensor(_x, dtype=DTYPE.FLOAT32, requires_grad=False)
        y = tensorgrad.Tensor(_y, dtype=DTYPE.INT32, requires_grad=False)
        m = tensorgrad.nn.Sequential(
            tensorgrad.nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=(3, 3),
                padding='same',
            ).init_from_torch(tm[0]),
            tensorgrad.nn.BatchNorm2d(4).init_from_torch(tm[1]),
            tensorgrad.nn.ReLU(),
            tensorgrad.nn.AvgPool2d((2, 2)),
            tensorgrad.nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=(3, 3),
                padding='same',
            ).init_from_torch(tm[4]),
            tensorgrad.nn.BatchNorm2d(8).init_from_torch(tm[5]),
            tensorgrad.nn.ReLU(),
            tensorgrad.nn.AvgPool2d((2, 2)),
            tensorgrad.nn.Flatten(),
            tensorgrad.nn.Linear(8 * 7 * 7, 10).init_from_torch(tm[-1]),
        )
        loss_fn = tensorgrad.nn.CrossEntropyLoss()
        optim = tensorgrad.optim.SGD(m.parameters(), 0.01)

        self.helper._train(
            tx, ty,
            tm, tloss_fn, toptim,
            x, y,
            m, loss_fn, optim,
            num_epochs,
            batch_size
        )


class Helper(unittest.TestCase):

    def _train(
        self,
        tx, ty,
        tm, tloss_fn, toptim,
        x, y,
        m, loss_fn, optim,
        num_epochs,
        batch_size,
    ):
        tparams = dict(tm.named_parameters())
        params = dict(m.named_parameters())
        self.assertEqual(tparams.keys(), params.keys(), f'params')
        
        tlosses = []
        losses = []
        check_tol = 1e-4
        step = 0
        for epoch in range(num_epochs):
            epoch += 1
            for i in range(0, len(tx), batch_size):
                step += 1
                
                txb = tx[i:i+batch_size]
                tyb = ty[i:i+batch_size]
                toptim.zero_grad()
                to = tm(txb)
                tloss = tloss_fn(to, tyb)
                tloss.backward()
                toptim.step()
                tlosses.append(tloss.item())
                
                xb = x[i:i+batch_size]
                yb = y[i:i+batch_size]
                optim.zero_grad()
                o = m(xb)
                loss = loss_fn(o, yb)
                loss.backward()
                optim.step()
                losses.append(loss.item())

                if VERBOSE:
                    if step == 1:
                        print()
                    tloss = np.mean(tlosses)
                    loss = np.mean(losses)
                    print(f'epoch: {epoch} step: {step} torch_loss: {tloss:.8f} tensorgrad_loss: {loss:.8f}')
                
                self._check_tensors([[tloss, loss, check_tol, 'loss']])
                for pname in params:
                    tp = tparams[pname]
                    p = params[pname]
                    self._check_tensors([
                        [tp, p, check_tol, f'{pname}@data'],
                        [tp.grad, p.grad, check_tol, f'{pname}@grad'],
                    ])

    def _check_tensors(self, pairs):
        for tt, t, tol, name in pairs:
            self.assertTrue(check_tensors(tt.tolist(), t.tolist(), tol=tol, show_diff=SHOW_DIFF), msg=name)
