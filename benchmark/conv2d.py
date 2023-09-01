import os
import time

import numpy as np
import torch
from tabulate import tabulate

import tensorgrad
from tensorgrad.ops.cpu.conv2d import Conv2D


N = int(os.getenv('NUM_RUNS', '500'))
RUNS = [
    [(4, 3, 224, 224), (3, 3), (1, 1), (0, 0)],
    [(4, 16, 112, 112), (3, 3), (1, 1), (0, 0)],
    [(4, 32, 54, 54), (3, 3), (1, 1), (0, 0)],
    [(4, 64, 28, 28), (3, 3), (1, 1), (0, 0)],
    [(4, 128, 14, 14), (3, 3), (1, 1), (0, 0)],
    [(4, 256, 7, 7), (3, 3), (1, 1), (0, 0)],
]


def forward(_x, _k, _b, stride, padding):
    times = []
    for _ in range(N):
        x = tensorgrad.Tensor(_x, requires_grad=False)
        k = tensorgrad.Tensor(_k, requires_grad=False)
        b = tensorgrad.Tensor(_b, requires_grad=False)
        op = Conv2D(x=x, kernel=k, bias=b, stride=stride, padding=padding)
        st = None
        ft = None
        st = time.perf_counter()
        op.forward()
        ft = time.perf_counter() - st
        times.append(ft)
    t = np.median(times[10:])
    return t


def backward(_x, _k, _b, stride, padding):
    times = []
    for _ in range(N):
        x = tensorgrad.Tensor(_x, requires_grad=True)
        k = tensorgrad.Tensor(_k, requires_grad=True)
        b = tensorgrad.Tensor(_b, requires_grad=True)
        op = Conv2D(x=x, kernel=k, bias=b, stride=stride, padding=padding)
        st = None
        ft = None
        o = op.forward()
        o._children = (x, k, b)
        o._op = op
        st = time.time()
        o.sum().backward()
        ft = time.time() - st
        times.append(ft)
    t = np.median(times[:10])
    return t


def torch_forward(_x, _k, _b, stride, padding):
    times = []
    for _ in range(N):
        x = torch.from_numpy(_x)
        k = torch.from_numpy(_k)
        b = torch.from_numpy(_b)
        st = None
        ft = None
        with torch.no_grad():
            st = time.perf_counter()
            torch.nn.functional.conv2d(x, k, b, stride=stride, padding=padding)
            ft = time.perf_counter() - st
            times.append(ft)
    t = np.median(times[10:])
    return t


def torch_backward(_x, _k, _b, stride, padding):
    times = []
    for _ in range(N):
        x = torch.from_numpy(_x)
        x.requires_grad = True
        k = torch.from_numpy(_k)
        k.requires_grad = True
        b = torch.from_numpy(_b)
        b.requires_grad = True
        st = None
        ft = None
        o = torch.nn.functional.conv2d(x, k, b, stride=stride, padding=padding)
        st = time.time()
        o.sum().backward()
        ft = time.time() - st
        times.append(ft)
    t = np.median(times[:10])
    return t


def main():
    forward_data = []
    backward_data = []
    
    for i, params in enumerate(RUNS):
        input_size, kernel_size, stride, padding = params
        batch_size = input_size[0]
        height, width = input_size[-2:]
        ci = co = input_size[1]
        kernel_size = (co, ci, *kernel_size)
        
        x = np.random.normal(size=input_size).astype('float32')
        k = np.random.normal(size=kernel_size).astype('float32')
        b = np.random.normal(size=(co,)).astype('float32')

        tftime = torch_forward(_x=x, _k=k, _b=b, stride=stride, padding=padding)
        ftime = forward(_x=x, _k=k, _b=b, stride=stride, padding=padding)
        fdiff = ftime / tftime
        forward_data.append({
            'batch': batch_size,
            'height': height,
            'width': width,
            'channels': ci,
            'kernel': kernel_size[2],
            'torch': round(tftime, 5),
            'tensorgrad': round(ftime, 5),
            'diff': round(fdiff, 5),
        })

        tbtime = torch_backward(_x=x, _k=k, _b=b, stride=stride, padding=padding)
        btime = backward(_x=x, _k=k, _b=b, stride=stride, padding=padding)
        bdiff = btime / tbtime
        backward_data.append({
            'batch': batch_size,
            'height': height,
            'width': width,
            'channels': ci,
            'kernel': kernel_size[2],
            'torch': round(tbtime, 5),
            'tensorgrad': round(btime, 5),
            'diff': round(bdiff, 5),
        })
        print(f'[{i+1}/{len(RUNS)}]: {params}')
    
    fdiff = np.mean([r['diff'] for r in forward_data])
    forward_data.append({
        'batch': '',
        'height': '',
        'width': '',
        'channels': '',
        'kernel': '',
        'torch': '',
        'tensorgrad': 'avg. diff',
        'diff': round(fdiff, 5),
    })
    bdiff = np.mean([r['diff'] for r in backward_data])
    backward_data.append({
        'batch': '',
        'height': '',
        'width': '',
        'channels': '',
        'kernel': '',
        'torch': '',
        'tensorgrad': 'avg. diff',
        'diff': round(bdiff, 5),
    })
    print()
    print('FORWARD:')
    print(tabulate(forward_data, headers="keys"))
    print()
    print('BACKWARD:')
    print(tabulate(backward_data, headers="keys"))



if __name__ == '__main__':
    main()
