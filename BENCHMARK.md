Benchmarks below compare performance of `tensorgrad` against `PyTorch`.  
The most interesting part about all this is the last column which shows how many times `tensorgrad` is slower than `PyTorch` given some `nn` module, size of input and device.

Specifically, for each `nn` module I measure time of forward and backward passes over inputs of different sizes using `PyTorch` and using `tensorgrad`. Each table contains results of a particular `nn` module's pass on a particular `tensorgrad` device. It's worth noting that `cuda` times of `tensorgrad` are compared with `cpu` times of `PyTorch` (because of using cpu-only build of `PyTorch`).


# nn.Linear
### CPU
#### Forward
| inputs                        |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-------------------------------|-----------------|------------------|---------------------------|
| input=32x64, features=128     |         1e-05   |          7e-05   |                   7.957   |
| input=64x1024, features=2048  |         0.00094 |          0.00333 |                   3.53146 |
| input=128x2048, features=4096 |         0.00826 |          0.02759 |                   3.34093 |
| overall                       |         0.00094 |          0.00333 |                   3.53146 |
#### Backward
| inputs                        |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-------------------------------|-----------------|------------------|---------------------------|
| input=32x64, features=128     |         6e-05   |          0.00013 |                   2.31789 |
| input=64x1024, features=2048  |         0.00297 |          0.02247 |                   7.56456 |
| input=128x2048, features=4096 |         0.02283 |          0.19425 |                   8.50893 |
| overall                       |         0.00297 |          0.02247 |                   7.56456 |
### CUDA
#### Forward
| inputs                        |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-------------------------------|-----------------|------------------|---------------------------|
| input=32x64, features=128     |         1e-05   |          0.00023 |                  19.7145  |
| input=64x1024, features=2048  |         0.00063 |          0.00025 |                   0.40021 |
| input=128x2048, features=4096 |         0.00679 |          0.00026 |                   0.03803 |
| overall                       |         0.00063 |          0.00025 |                   0.40021 |
#### Backward
| inputs                        |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-------------------------------|-----------------|------------------|---------------------------|
| input=32x64, features=128     |         7e-05   |          0.00057 |                   8.50881 |
| input=64x1024, features=2048  |         0.00253 |          0.00062 |                   0.24498 |
| input=128x2048, features=4096 |         0.0223  |          0.00061 |                   0.02735 |
| overall                       |         0.00253 |          0.00061 |                   0.24498 |
# nn.Conv2d
### CPU
#### Forward
| inputs                                   |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|------------------------------------------|-----------------|------------------|---------------------------|
| input=4x3x32x32 kernel=3 channels=16     |         5e-05   |          0.00027 |                   5.28848 |
| input=8x32x64x64 kernel=3 channels=64    |         0.00424 |          0.02066 |                   4.87644 |
| input=8x64x128x128 kernel=3 channels=128 |         0.06646 |          0.17148 |                   2.58034 |
| overall                                  |         0.00424 |          0.02066 |                   4.87644 |
#### Backward
| inputs                                   |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|------------------------------------------|-----------------|------------------|---------------------------|
| input=4x3x32x32 kernel=3 channels=16     |         0.00014 |          0.0014  |                   9.82634 |
| input=8x32x64x64 kernel=3 channels=64    |         0.00706 |          0.09186 |                  13.0172  |
| input=8x64x128x128 kernel=3 channels=128 |         0.06874 |          0.71355 |                  10.3801  |
| overall                                  |         0.00706 |          0.09186 |                  10.3801  |
### CUDA
#### Forward
| inputs                                   |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|------------------------------------------|-----------------|------------------|---------------------------|
| input=4x3x32x32 kernel=3 channels=16     |         6e-05   |          0.00064 |                  10.2036  |
| input=8x32x64x64 kernel=3 channels=64    |         0.00285 |          0.00066 |                   0.23203 |
| input=8x64x128x128 kernel=3 channels=128 |         0.06073 |          0.00498 |                   0.08204 |
| overall                                  |         0.00285 |          0.00066 |                   0.23203 |
#### Backward
| inputs                                   |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|------------------------------------------|-----------------|------------------|---------------------------|
| input=4x3x32x32 kernel=3 channels=16     |         0.00016 |          0.00156 |                   9.50008 |
| input=8x32x64x64 kernel=3 channels=64    |         0.00656 |          0.00316 |                   0.48117 |
| input=8x64x128x128 kernel=3 channels=128 |         0.08608 |          0.02187 |                   0.25411 |
| overall                                  |         0.00656 |          0.00316 |                   0.48117 |
# nn.MultiheadAttention
### CPU
#### Forward
| inputs                                 |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|----------------------------------------|-----------------|------------------|---------------------------|
| input=16x32x64 d_model=64 n_heads=4    |         0.0002  |          0.00175 |                   8.84703 |
| input=32x64x128 d_model=128 n_heads=8  |         0.00229 |          0.01677 |                   7.32494 |
| input=32x128x256 d_model=256 n_heads=8 |         0.0148  |          0.07684 |                   5.19165 |
| overall                                |         0.00229 |          0.01677 |                   7.32494 |
#### Backward
| inputs                                 |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|----------------------------------------|-----------------|------------------|---------------------------|
| input=16x32x64 d_model=64 n_heads=4    |         0.0004  |          0.00745 |                   18.7644 |
| input=32x64x128 d_model=128 n_heads=8  |         0.00386 |          0.2466  |                   63.8829 |
| input=32x128x256 d_model=256 n_heads=8 |         0.01779 |          1.57431 |                   88.5085 |
| overall                                |         0.00386 |          0.2466  |                   63.8829 |
### CUDA
#### Forward
| inputs                                 |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|----------------------------------------|-----------------|------------------|---------------------------|
| input=16x32x64 d_model=64 n_heads=4    |         0.00023 |          0.00191 |                   8.25529 |
| input=32x64x128 d_model=128 n_heads=8  |         0.0012  |          0.00196 |                   1.62443 |
| input=32x128x256 d_model=256 n_heads=8 |         0.01626 |          0.00193 |                   0.11856 |
| overall                                |         0.0012  |          0.00193 |                   1.62443 |
#### Backward
| inputs                                 |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|----------------------------------------|-----------------|------------------|---------------------------|
| input=16x32x64 d_model=64 n_heads=4    |         0.00038 |          0.0026  |                   6.88775 |
| input=32x64x128 d_model=128 n_heads=8  |         0.00485 |          0.0026  |                   0.53555 |
| input=32x128x256 d_model=256 n_heads=8 |         0.01729 |          0.00274 |                   0.15873 |
| overall                                |         0.00485 |          0.0026  |                   0.53555 |
# nn.Softmax
### CPU
#### Forward
| inputs                |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-----------------------|-----------------|------------------|---------------------------|
| input=128x64 dim=-1   |           0     |          5e-05   |                   9.71885 |
| input=512x256 dim=-1  |           2e-05 |          0.00028 |                  17.1162  |
| input=1024x512 dim=-1 |           7e-05 |          0.00142 |                  21.2956  |
| overall               |           2e-05 |          0.00028 |                  17.1162  |
#### Backward
| inputs                |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-----------------------|-----------------|------------------|---------------------------|
| input=128x64 dim=-1   |         3e-05   |          0.00073 |                   21.8265 |
| input=512x256 dim=-1  |         7e-05   |          0.09498 |                 1431.73   |
| input=1024x512 dim=-1 |         0.00015 |          0.66259 |                 4488.73   |
| overall               |         7e-05   |          0.09498 |                 1431.73   |
### CUDA
#### Forward
| inputs                |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-----------------------|-----------------|------------------|---------------------------|
| input=128x64 dim=-1   |         0.00771 |           0.0001 |                   0.01325 |
| input=512x256 dim=-1  |         3e-05   |           0.0001 |                   3.88199 |
| input=1024x512 dim=-1 |         9e-05   |           0.0001 |                   1.09995 |
| overall               |         9e-05   |           0.0001 |                   1.09995 |
#### Backward
| inputs                |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-----------------------|-----------------|------------------|---------------------------|
| input=128x64 dim=-1   |         4e-05   |          0.00052 |                  14.7455  |
| input=512x256 dim=-1  |         7e-05   |          0.00056 |                   7.69522 |
| input=1024x512 dim=-1 |         0.00036 |          0.0006  |                   1.64598 |
| overall               |         7e-05   |          0.00056 |                   7.69522 |