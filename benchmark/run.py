from .const import SUITE, RUN_ON_CUDA, NUM_RUNS, MARKDOWN_PATH, INTRO
from .runner import BenchmarkRunner


def main():
    buffer = [INTRO, '\n']
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
        buffer.append(f'### CPU')
        buffer.append(f'#### Forward')
        buffer.append(mod_cpu_forward)
        buffer.append(f'#### Backward')
        buffer.append(mod_cpu_backward)
        if RUN_ON_CUDA:
            buffer.append(f'### CUDA')
            buffer.append(f'#### Forward')
            buffer.append(mod_cuda_forward)
            buffer.append(f'#### Backward')
            buffer.append(mod_cuda_backward)
    data = '\n'.join(buffer)
    print('-> Done.')
    print(data)
    with open(MARKDOWN_PATH, 'w') as f:
        f.write(data) 


if __name__ == '__main__':
    main()
