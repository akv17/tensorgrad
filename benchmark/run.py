import os

import torch
import tensorgrad

from .runner import BenchmarkRunner

MARKDOWN_PATH = os.getenv('MD_PATH', 'BENCHMARK.md')
NUM_RUNS = int(os.getenv('NUM_RUNS', '10').strip())
RUN_ON_CUDA = os.getenv('RUN_ON_CUDA') == '1'
SUITE = {
    'nn.Linear': {
        'input=32x64, features=128': {
            'torch': {
                'args': (torch.rand(32, 64),),
                'module': torch.nn.Linear(64, 128),
            },
            'tensorgrad': {
                'args': (tensorgrad.tensor.rand(32, 64),),
                'module': tensorgrad.nn.Linear(64, 128),
            },
        },
        'input=64x1024, features=2048': {
            'torch': {
                'args': (torch.rand(32, 1024),),
                'module': torch.nn.Linear(1024, 2048),
            },
            'tensorgrad': {
                'args': (tensorgrad.tensor.rand(32, 1024),),
                'module': tensorgrad.nn.Linear(1024, 2048),
            },
        },
        'input=128x2048, features=4096': {
            'torch': {
                'args': (torch.rand(128, 2048),),
                'module': torch.nn.Linear(2048, 4096),
            },
            'tensorgrad': {
                'args': (tensorgrad.tensor.rand(128, 2048),),
                'module': tensorgrad.nn.Linear(2048, 4096),
            },
        },
    },
    'nn.Conv2d': {
        'input=4x3x32x32 kernel=3 channels=16': {
            'torch': {
                'args': (torch.rand(4, 3, 32, 32),),
                'module': torch.nn.Conv2d(3, 16, 3),
            },
            'tensorgrad': {
                'args': (tensorgrad.tensor.rand(4, 3, 32, 32),),
                'module': tensorgrad.nn.Conv2d(3, 16, 3),
            }
        },
        'input=8x32x64x64 kernel=3 channels=64': {
            'torch': {
                'args': (torch.rand(8, 32, 64, 64),),
                'module': torch.nn.Conv2d(32, 64, 3),
            },
            'tensorgrad': {
                'args': (tensorgrad.tensor.rand(8, 32, 64, 64),),
                'module': tensorgrad.nn.Conv2d(32, 64, 3),
            }
        },
        'input=8x64x128x128 kernel=3 channels=128': {
            'torch': {
                'args': (torch.rand(8, 64, 128, 128),),
                'module': torch.nn.Conv2d(64, 128, 3),
            },
            'tensorgrad': {
                'args': (tensorgrad.tensor.rand(8, 64, 128, 128),),
                'module': tensorgrad.nn.Conv2d(64, 128, 3),
            }
        },
    },
    'nn.MultiheadAttention': {
        'input=16x32x64 d_model=64 n_heads=4': {
            'torch': {
                'args': (
                    torch.rand(16, 32, 64),
                    torch.rand(16, 32, 64),
                    torch.rand(16, 32, 64),
                ),
                'module': torch.nn.MultiheadAttention(embed_dim=64, num_heads=4)
            },
            'tensorgrad': {
                'args': (
                    tensorgrad.tensor.rand(16, 32, 64),
                    tensorgrad.tensor.rand(16, 32, 64),
                    tensorgrad.tensor.rand(16, 32, 64),
                ),
                'module': tensorgrad.nn.MultiheadAttention(embed_dim=64, num_heads=4)
            }
        },
        'input=32x64x128 d_model=128 n_heads=8': {
            'torch': {
                'args': (
                    torch.rand(32, 64, 128),
                    torch.rand(32, 64, 128),
                    torch.rand(32, 64, 128),
                ),
                'module': torch.nn.MultiheadAttention(embed_dim=128, num_heads=8)
            },
            'tensorgrad': {
                'args': (
                    tensorgrad.tensor.rand(32, 64, 128),
                    tensorgrad.tensor.rand(32, 64, 128),
                    tensorgrad.tensor.rand(32, 64, 128),
                ),
                'module': tensorgrad.nn.MultiheadAttention(embed_dim=128, num_heads=8)
            }
        },
        'input=32x128x256 d_model=256 n_heads=8': {
            'torch': {
                'args': (
                    torch.rand(32, 128, 256),
                    torch.rand(32, 128, 256),
                    torch.rand(32, 128, 256),
                ),
                'module': torch.nn.MultiheadAttention(embed_dim=256, num_heads=8)
            },
            'tensorgrad': {
                'args': (
                    tensorgrad.tensor.rand(32, 128, 256),
                    tensorgrad.tensor.rand(32, 128, 256),
                    tensorgrad.tensor.rand(32, 128, 256),
                ),
                'module': tensorgrad.nn.MultiheadAttention(embed_dim=256, num_heads=8)
            }
        },
    },
    'nn.Softmax': {
        'input=128x64 dim=-1': {
            'torch': {
                'args': (torch.rand(128, 64, requires_grad=True),),
                'module': torch.nn.Softmax(-1),
            },
            'tensorgrad': {
                'args': (tensorgrad.tensor.rand(128, 64),),
                'module': tensorgrad.nn.Softmax(-1),
            },
        },
        'input=512x256 dim=-1': {
            'torch': {
                'args': (torch.rand(512, 256, requires_grad=True),),
                'module': torch.nn.Softmax(-1),
            },
            'tensorgrad': {
                'args': (tensorgrad.tensor.rand(512, 256),),
                'module': tensorgrad.nn.Softmax(-1),
            },
        },
        'input=1024x512 dim=-1': {
            'torch': {
                'args': (torch.rand(1024, 512, requires_grad=True),),
                'module': torch.nn.Softmax(-1),
            },
            'tensorgrad': {
                'args': (tensorgrad.tensor.rand(1024, 512),),
                'module': tensorgrad.nn.Softmax(-1),
            },
        },
    }
}


def main():
    buffer = []
    for mod, mod_suite in SUITE.items():
        print(f'-> Running {mod} on CPU')
        cpu_runner = BenchmarkRunner(num_runs=NUM_RUNS, suite=mod_suite, device='cpu')
        mod_cpu_data = cpu_runner.run()
        mod_cpu_forward = mod_cpu_data['forward']
        mod_cpu_backward = mod_cpu_data['backward']
        if RUN_ON_CUDA:
            print(f'-> Running {mod} on CUDA')
            cuda_runner = BenchmarkRunner(num_runs=NUM_RUNS, suite=mod_suite, device='cuda')
            mod_cuda_data = cuda_runner.run()
            mod_cuda_forward = mod_cuda_data['forward']
            mod_cuda_backward = mod_cuda_data['backward']
        buffer.append(f'# {mod}')
        buffer.append(f'## CPU')
        buffer.append(f'### Forward')
        buffer.append(mod_cpu_forward)
        buffer.append(f'### Backward')
        buffer.append(mod_cpu_backward)
        if RUN_ON_CUDA:
            buffer.append(f'## CUDA')
            buffer.append(f'### Forward')
            buffer.append(mod_cuda_forward)
            buffer.append(f'### Backward')
            buffer.append(mod_cuda_backward)
    data = '\n'.join(buffer)
    print('-> Done.')
    print(data)
    with open(MARKDOWN_PATH, 'w') as f:
        f.write(data) 


if __name__ == '__main__':
    main()
