import time
import statistics

import torch
import tensorgrad
from tabulate import tabulate


class BenchmarkRunner:

    def __init__(
        self,
        num_runs,
        suite,
        device='cpu',
    ):
        self.num_runs = num_runs
        self.suite = suite
        self.stats = {
            'forward': {},
            'backward': {},
        }
        self.device = device
        self.data = None
    
    def run(self):
        self._run()
        self._summarize()
        return self.data

    def _run(self):
        for key, suite in self.suite.items():
            self._forward_torch(key=key, suite=suite)
            self._backward_torch(key=key, suite=suite)
            self._forward_tensorgrad(key=key, suite=suite)
            self._backward_tensorgrad(key=key, suite=suite)
    
    def _summarize(self):
        forward = self._generate_summary_table('forward')
        backward = self._generate_summary_table('backward')
        data = {'forward': forward, 'backward': backward}
        self.data = data

    def _generate_summary_table(self, key):
        stats = self.stats[key]
        records = []
        for key, data in stats.items():
            torch_time = data['torch']
            tensorgrad_time = data['tensorgrad']
            diff = tensorgrad_time / torch_time
            records.append({
                'inputs': key,
                'torch cpu, s.': round(torch_time, 5),
                'tensorgrad, s.': round(tensorgrad_time, 5),
                'tensorgrad is slower by': round(diff, 5),
            })
        records.append({
            'inputs': 'overall',
            'torch cpu, s.': round(statistics.median([r['torch cpu, s.'] for r in records[:]]), 5),
            'tensorgrad, s.': round(statistics.median([r['tensorgrad, s.'] for r in records[:]]), 5),
            'tensorgrad is slower by': round(statistics.median([r['tensorgrad is slower by'] for r in records[:]]), 5),
        })
        table = tabulate(records, headers='keys', tablefmt='github')
        return table

    def _forward_torch(self, key, suite):
        x = suite['torch']['args']
        mod = suite['torch']['module']
        times = []
        mod(*x)
        with torch.no_grad():
            for _ in range(self.num_runs):
                start = time.perf_counter()
                mod(*x)
                end = time.perf_counter() - start
                times.append(end)
        time_ = statistics.median(times)
        self.stats['forward'].setdefault(key, {})['torch'] = time_
        return time_
        
    def _backward_torch(self, key, suite):
        x = suite['torch']['args']
        mod = suite['torch']['module']
        times = []
        mod(*x)
        for _ in range(self.num_runs):
            o = mod(*x)
            o = o[0] if isinstance(o, tuple) else o
            start = time.perf_counter()
            o.sum().backward()
            end = time.perf_counter() - start
            times.append(end)
        time_ = statistics.median(times)
        self.stats['backward'].setdefault(key, {})['torch'] = time_
    
    def _forward_tensorgrad(self, key, suite):
        device = self.device
        x = suite['tensorgrad']['args']
        x = tuple(xi.to(device) for xi in x)
        mod = suite['tensorgrad']['module']
        mod.to(device)
        times = []
        mod(*x)
        with tensorgrad.no_grad():
            for _ in range(self.num_runs):
                start = time.perf_counter()
                mod(*x)
                end = time.perf_counter() - start
                times.append(end)
        time_ = statistics.median(times)
        self.stats['forward'].setdefault(key, {})['tensorgrad'] = time_
    
    def _backward_tensorgrad(self, key, suite):
        device = self.device
        x = suite['tensorgrad']['args']
        x = tuple(xi.to(device) for xi in x)
        mod = suite['tensorgrad']['module']
        mod.to(device)
        times = []
        mod(*x)
        for _ in range(self.num_runs):
            o = mod(*x)
            start = time.perf_counter()
            o.sum().backward()
            end = time.perf_counter() - start
            times.append(end)
        time_ = statistics.median(times)
        self.stats['backward'].setdefault(key, {})['tensorgrad'] = time_
