# About
Benchmarks below compare performance of `tensorgrad` against `PyTorch`.  
The most interesting part about all this is the last column which shows how many times `tensorgrad` is slower than `PyTorch` given some `nn` module, size of input and device.

Specifically, for each `nn` module I measure time of forward and backward passes over inputs of different sizes using `PyTorch` and using `tensorgrad` on different devices. Each table contains results of a particular `nn` module's pass on a particular `tensorgrad` device. It's worth noting that `cuda` times of `tensorgrad` are compared with `cpu` times of `PyTorch` (because of using cpu-only build of `PyTorch`).


# nn.Linear
### CPU
#### Forward
| inputs                        |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-------------------------------|-----------------|------------------|---------------------------|
| input=32x64, features=128     |         1e-05   |          7e-05   |                   8.80803 |
| input=64x1024, features=2048  |         0.0004  |          0.00426 |                  10.6517  |
| input=128x2048, features=4096 |         0.00859 |          0.01457 |                   1.6971  |
| overall                       |         0.0004  |          0.00426 |                   8.80803 |
#### Backward
| inputs                        |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-------------------------------|-----------------|------------------|---------------------------|
| input=32x64, features=128     |         6e-05   |          0.00012 |                   2.2049  |
| input=64x1024, features=2048  |         0.00264 |          0.02264 |                   8.59082 |
| input=128x2048, features=4096 |         0.02278 |          0.20747 |                   9.1081  |
| overall                       |         0.00264 |          0.02264 |                   8.59082 |
### CUDA
#### Forward
| inputs                        |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-------------------------------|-----------------|------------------|---------------------------|
| input=32x64, features=128     |         1e-05   |          0.00024 |                  27.1065  |
| input=64x1024, features=2048  |         0.0004  |          0.00026 |                   0.64083 |
| input=128x2048, features=4096 |         0.00575 |          0.00025 |                   0.04421 |
| overall                       |         0.0004  |          0.00025 |                   0.64083 |
#### Backward
| inputs                        |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-------------------------------|-----------------|------------------|---------------------------|
| input=32x64, features=128     |         6e-05   |          0.00055 |                   9.88692 |
| input=64x1024, features=2048  |         0.00244 |          0.00061 |                   0.24892 |
| input=128x2048, features=4096 |         0.02253 |          0.00063 |                   0.02808 |
| overall                       |         0.00244 |          0.00061 |                   0.24892 |
# nn.Conv2d
### CPU
#### Forward
| inputs                                   |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|------------------------------------------|-----------------|------------------|---------------------------|
| input=4x3x32x32 kernel=3 channels=16     |         4e-05   |          0.00028 |                   6.29908 |
| input=8x32x64x64 kernel=3 channels=64    |         0.00415 |          0.02087 |                   5.03028 |
| input=8x64x128x128 kernel=3 channels=128 |         0.04715 |          0.16769 |                   3.55654 |
| overall                                  |         0.00415 |          0.02087 |                   5.03028 |
#### Backward
| inputs                                   |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|------------------------------------------|-----------------|------------------|---------------------------|
| input=4x3x32x32 kernel=3 channels=16     |         0.00014 |          0.00143 |                  10.4189  |
| input=8x32x64x64 kernel=3 channels=64    |         0.00789 |          0.09247 |                  11.724   |
| input=8x64x128x128 kernel=3 channels=128 |         0.08688 |          0.64717 |                   7.44874 |
| overall                                  |         0.00789 |          0.09247 |                  10.4189  |
### CUDA
#### Forward
| inputs                                   |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|------------------------------------------|-----------------|------------------|---------------------------|
| input=4x3x32x32 kernel=3 channels=16     |         7e-05   |          0.00062 |                   9.53919 |
| input=8x32x64x64 kernel=3 channels=64    |         0.00316 |          0.00066 |                   0.20947 |
| input=8x64x128x128 kernel=3 channels=128 |         0.04744 |          0.00458 |                   0.09658 |
| overall                                  |         0.00316 |          0.00066 |                   0.20947 |
#### Backward
| inputs                                   |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|------------------------------------------|-----------------|------------------|---------------------------|
| input=4x3x32x32 kernel=3 channels=16     |         0.00016 |          0.00153 |                   9.42888 |
| input=8x32x64x64 kernel=3 channels=64    |         0.00728 |          0.00307 |                   0.42235 |
| input=8x64x128x128 kernel=3 channels=128 |         0.0691  |          0.02135 |                   0.309   |
| overall                                  |         0.00728 |          0.00307 |                   0.42235 |
# nn.MultiheadAttention
### CPU
#### Forward
| inputs                                 |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|----------------------------------------|-----------------|------------------|---------------------------|
| input=16x32x64 d_model=64 n_heads=4    |         0.00018 |          0.00126 |                   7.11561 |
| input=32x64x128 d_model=128 n_heads=8  |         0.001   |          0.01312 |                  13.103   |
| input=32x128x256 d_model=256 n_heads=8 |         0.0126  |          0.05018 |                   3.98075 |
| overall                                |         0.001   |          0.01312 |                   7.11561 |
#### Backward
| inputs                                 |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|----------------------------------------|-----------------|------------------|---------------------------|
| input=16x32x64 d_model=64 n_heads=4    |         0.00035 |          0.00527 |                   15.2376 |
| input=32x64x128 d_model=128 n_heads=8  |         0.00246 |          0.23274 |                   94.6321 |
| input=32x128x256 d_model=256 n_heads=8 |         0.01583 |          1.5761  |                   99.5361 |
| overall                                |         0.00246 |          0.23274 |                   94.6321 |
### CUDA
#### Forward
| inputs                                 |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|----------------------------------------|-----------------|------------------|---------------------------|
| input=16x32x64 d_model=64 n_heads=4    |         0.00025 |          0.00255 |                  10.349   |
| input=32x64x128 d_model=128 n_heads=8  |         0.00102 |          0.00192 |                   1.87972 |
| input=32x128x256 d_model=256 n_heads=8 |         0.01074 |          0.0019  |                   0.17737 |
| overall                                |         0.00102 |          0.00192 |                   1.87972 |
#### Backward
| inputs                                 |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|----------------------------------------|-----------------|------------------|---------------------------|
| input=16x32x64 d_model=64 n_heads=4    |         0.00047 |          0.00254 |                   5.42503 |
| input=32x64x128 d_model=128 n_heads=8  |         0.0027  |          0.00259 |                   0.96005 |
| input=32x128x256 d_model=256 n_heads=8 |         0.01579 |          0.00265 |                   0.16774 |
| overall                                |         0.0027  |          0.00259 |                   0.96005 |
# nn.Softmax
### CPU
#### Forward
| inputs                |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-----------------------|-----------------|------------------|---------------------------|
| input=128x64 dim=-1   |         0       |          5e-05   |                   9.62487 |
| input=512x256 dim=-1  |         2e-05   |          0.0003  |                  18.9474  |
| input=1024x512 dim=-1 |         0.00859 |          0.00082 |                   0.09526 |
| overall               |         2e-05   |          0.0003  |                   9.62487 |
#### Backward
| inputs                |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-----------------------|-----------------|------------------|---------------------------|
| input=128x64 dim=-1   |         3e-05   |          0.00072 |                   21.2054 |
| input=512x256 dim=-1  |         8e-05   |          0.09306 |                 1156.66   |
| input=1024x512 dim=-1 |         0.00034 |          0.65907 |                 1913.83   |
| overall               |         8e-05   |          0.09306 |                 1156.66   |
### CUDA
#### Forward
| inputs                |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-----------------------|-----------------|------------------|---------------------------|
| input=128x64 dim=-1   |         0.00878 |           0.0001 |                   0.01167 |
| input=512x256 dim=-1  |         3e-05   |           0.0001 |                   3.01834 |
| input=1024x512 dim=-1 |         0.0001  |           0.0001 |                   0.99662 |
| overall               |         0.0001  |           0.0001 |                   0.99662 |
#### Backward
| inputs                |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-----------------------|-----------------|------------------|---------------------------|
| input=128x64 dim=-1   |         4e-05   |          0.00055 |                  13.273   |
| input=512x256 dim=-1  |         7e-05   |          0.00056 |                   7.66853 |
| input=1024x512 dim=-1 |         0.00021 |          0.0006  |                   2.79397 |
| overall               |         7e-05   |          0.00056 |                   7.66853 |