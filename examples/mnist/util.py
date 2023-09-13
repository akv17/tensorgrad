import sys

try:
    import torchvision
except ImportError:
    msg = 'torchvision is required to train MNIST'
    raise Exception(msg)

import numpy as np

import tensorgrad


class Dataset:

    def __init__(self, is_train=True, truncate=None, root='.datasets'):
        self.is_train = is_train
        self.truncate = truncate
        self.dataset = torchvision.datasets.MNIST(root=root, download=True, train=is_train)
        self.size = self.truncate or len(self.dataset)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image).astype('float32')
        image /= 255.0
        image = tensorgrad.tensor(image)
        # add channel dim.
        image = image.unsqueeze(0)
        return image, label


class DataLoader:

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    @property
    def steps_per_epoch(self):
        n = len(self.dataset) // self.batch_size + 1
        return n

    def __iter__(self):
        x = []
        y = []
        for i in range(len(self.dataset)):
            xi, yi = self.dataset[i]
            x.append(xi)
            y.append(yi)
            if len(x) == self.batch_size:
                x = self._collate_x(x)
                y = self._collate_y(y)
                yield x, y
                x = []
                y = []
        if x:
            x = self._collate_x(x)
            y = self._collate_y(y)
            yield x, y

    def _collate_x(self, tensors):
        with tensorgrad.no_grad():
            tensors = [t.unsqueeze(0) for t in tensors]
            out = tensors[0].concat(tensors[1:], 0)
        return out
    
    def _collate_y(self, values):
        with tensorgrad.no_grad():
            out = tensorgrad.tensor(values).long()
        return out


class Trainer:

    def __init__(
        self,
        dataloader,
        model,
        loss_fn,
        optimizer,
        device,
        epochs,
        evaluator=None,
    ):
        self.dataloader = dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.evaluator = evaluator
    
    def run(self):
        self.model.train()
        self.model.to(self.device)
        losses = []
        step = 0
        steps_per_epoch = self.dataloader.steps_per_epoch
        for epoch in range(self.epochs):
            epoch += 1
            for x, y in ProgressBar(self.dataloader, steps_per_epoch, prefix=f'Epoch: {epoch} '):
                step += 1
                inputs = x.to(self.device)
                targets = y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                loss_val = loss.detach().cpu().item()
                losses.append(loss_val)
            if self.evaluator is not None:
                loss_train = tensorgrad.tensor(losses).mean().item()
                self.model.eval()
                metrics = self.evaluator.run()
                self.model.train()
                print(f'Train Loss: {loss_train}')
                print('Metrics:')
                print(metrics, end='\n\n')


class ProgressBar:

    def __init__(self, iter, n, prefix=None, bins=25):
        self.it = iter
        self.n = n
        self.prefix = prefix or ''
        self.bins = bins
        self._stream = sys.stdout
    
    def __iter__(self):
        self._stream.flush()
        bins = [' '] * self.bins
        for i, it in enumerate(self.it):
            pos = self._calculate_bin(i)
            bins[pos] = '='
            display = ''.join(bins)
            percent = int((i+1) / self.n * 100)
            display = f'{self.prefix}[{display}] {percent}%'
            self._stream.write('\r')
            self._stream.write(display)
            yield it
        self._stream.write('\n')

    def _calculate_bin(self, i):
        return max(0, int(((i+1) / self.n) * self.bins) - 1)


class Evaluator:

    def __init__(self, dataloader, model, device):
        self.dataloader = dataloader
        self.model = model
        self.device = device
    
    def run(self):
        pred, true = self._forward()
        summary = self._summarize(pred, true)
        report = self._format(summary)
        return report

    def _forward(self):
        pred = []
        true = []
        with tensorgrad.no_grad():
            for x, y in self.dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                outputs = outputs.numpy()
                outputs = outputs.argmax(-1)
                pred.extend(outputs.tolist())
                true.extend(y.tolist())
        return pred, true

    def _summarize(self, pred, true):
        by_label = {}
        for p, t in zip(pred, true):
            is_correct = p == t
            by_label.setdefault(t, []).append(is_correct)
        by_label = {k: (tensorgrad.tensor(v).mean().item(), len(v)) for k, v in by_label.items()}
        by_label = {k: by_label[k] for k in sorted(by_label)}
        by_label['Total'] = (tensorgrad.tensor([v[0] for v in by_label.values()]).mean().item(), len(true))
        return by_label
    
    def _format(self, summary):
        buffer = [
            'Class\tAccuracy\tSize',
            *[f'{k}\t{v[0]:.4f}\t\t{v[1]}' for k, v in summary.items()]
        ]
        report = '\n'.join(buffer)
        return report
