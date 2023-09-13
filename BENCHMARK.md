# nn.Linear
## CPU
### Forward
| inputs                        |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-------------------------------|-----------------|------------------|---------------------------|
| input=32x64, features=128     |         1e-05   |          7e-05   |                   8.56341 |
| input=64x1024, features=2048  |         0.00041 |          0.00338 |                   8.29164 |
| input=128x2048, features=4096 |         0.00703 |          0.01471 |                   2.09388 |
| overall                       |         0.00041 |          0.00338 |                   8.29164 |
### Backward
| inputs                        |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-------------------------------|-----------------|------------------|---------------------------|
| input=32x64, features=128     |         5e-05   |          0.00012 |                   2.27142 |
| input=64x1024, features=2048  |         0.00279 |          0.02218 |                   7.96326 |
| input=128x2048, features=4096 |         0.02344 |          0.19212 |                   8.19792 |
| overall                       |         0.00279 |          0.02218 |                   7.96326 |
## CUDA
### Forward
| inputs                        |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-------------------------------|-----------------|------------------|---------------------------|
| input=32x64, features=128     |         1e-05   |          0.00025 |                  28.6056  |
| input=64x1024, features=2048  |         0.00074 |          0.00025 |                   0.33555 |
| input=128x2048, features=4096 |         0.00874 |          0.00027 |                   0.03081 |
| overall                       |         0.00074 |          0.00025 |                   0.33555 |
### Backward
| inputs                        |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-------------------------------|-----------------|------------------|---------------------------|
| input=32x64, features=128     |         6e-05   |          0.00058 |                  10.0899  |
| input=64x1024, features=2048  |         0.00257 |          0.00064 |                   0.25073 |
| input=128x2048, features=4096 |         0.02738 |          0.00069 |                   0.02537 |
| overall                       |         0.00257 |          0.00064 |                   0.25073 |
# nn.Conv2d
## CPU
### Forward
| inputs                                   |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|------------------------------------------|-----------------|------------------|---------------------------|
| input=4x3x32x32 kernel=3 channels=16     |         5e-05   |          0.00031 |                   6.10895 |
| input=8x32x64x64 kernel=3 channels=64    |         0.00564 |          0.0262  |                   4.64408 |
| input=8x64x128x128 kernel=3 channels=128 |         0.06854 |          0.17304 |                   2.5247  |
| overall                                  |         0.00564 |          0.0262  |                   4.64408 |
### Backward
| inputs                                   |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|------------------------------------------|-----------------|------------------|---------------------------|
| input=4x3x32x32 kernel=3 channels=16     |         0.00017 |          0.00179 |                   10.8117 |
| input=8x32x64x64 kernel=3 channels=64    |         0.00908 |          0.09298 |                   10.2378 |
| input=8x64x128x128 kernel=3 channels=128 |         0.07209 |          0.7665  |                   10.6323 |
| overall                                  |         0.00908 |          0.09298 |                   10.6323 |
## CUDA
### Forward
| inputs                                   |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|------------------------------------------|-----------------|------------------|---------------------------|
| input=4x3x32x32 kernel=3 channels=16     |         9e-05   |          0.0005  |                   5.80718 |
| input=8x32x64x64 kernel=3 channels=64    |         0.00611 |          0.00067 |                   0.1102  |
| input=8x64x128x128 kernel=3 channels=128 |         0.06617 |          0.00475 |                   0.07175 |
| overall                                  |         0.00611 |          0.00067 |                   0.1102  |
### Backward
| inputs                                   |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|------------------------------------------|-----------------|------------------|---------------------------|
| input=4x3x32x32 kernel=3 channels=16     |         0.00016 |          0.00121 |                   7.78375 |
| input=8x32x64x64 kernel=3 channels=64    |         0.01013 |          0.0031  |                   0.30608 |
| input=8x64x128x128 kernel=3 channels=128 |         0.08209 |          0.02117 |                   0.25794 |
| overall                                  |         0.01013 |          0.0031  |                   0.30608 |
# nn.MultiheadAttention
## CPU
### Forward
| inputs                                 |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|----------------------------------------|-----------------|------------------|---------------------------|
| input=16x32x64 d_model=64 n_heads=4    |         0.00021 |          0.00165 |                   7.79378 |
| input=32x64x128 d_model=128 n_heads=8  |         0.00115 |          0.01381 |                  12.049   |
| input=32x128x256 d_model=256 n_heads=8 |         0.01379 |          0.05508 |                   3.995   |
| overall                                |         0.00115 |          0.01381 |                   7.79378 |
### Backward
| inputs                                 |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|----------------------------------------|-----------------|------------------|---------------------------|
| input=16x32x64 d_model=64 n_heads=4    |         0.00039 |          0.00639 |                   16.22   |
| input=32x64x128 d_model=128 n_heads=8  |         0.00246 |          0.23956 |                   97.5042 |
| input=32x128x256 d_model=256 n_heads=8 |         0.01717 |          1.74274 |                  101.483  |
| overall                                |         0.00246 |          0.23956 |                   97.5042 |
## CUDA
### Forward
| inputs                                 |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|----------------------------------------|-----------------|------------------|---------------------------|
| input=16x32x64 d_model=64 n_heads=4    |         0.00024 |          0.00195 |                   8.14675 |
| input=32x64x128 d_model=128 n_heads=8  |         0.00117 |          0.002   |                   1.71549 |
| input=32x128x256 d_model=256 n_heads=8 |         0.01651 |          0.0021  |                   0.12722 |
| overall                                |         0.00117 |          0.002   |                   1.71549 |
### Backward
| inputs                                 |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|----------------------------------------|-----------------|------------------|---------------------------|
| input=16x32x64 d_model=64 n_heads=4    |         0.00044 |          0.00267 |                   6.12977 |
| input=32x64x128 d_model=128 n_heads=8  |         0.00346 |          0.00273 |                   0.78899 |
| input=32x128x256 d_model=256 n_heads=8 |         0.02214 |          0.00395 |                   0.17859 |
| overall                                |         0.00346 |          0.00273 |                   0.78899 |
# nn.Softmax
## CPU
### Forward
| inputs                |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-----------------------|-----------------|------------------|---------------------------|
| input=128x64 dim=-1   |         1e-05   |          7e-05   |                   8.2931  |
| input=512x256 dim=-1  |         4e-05   |          0.00036 |                   9.19821 |
| input=1024x512 dim=-1 |         0.00011 |          0.00156 |                  14.7386  |
| overall               |         4e-05   |          0.00036 |                   9.19821 |
### Backward
| inputs                |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-----------------------|-----------------|------------------|---------------------------|
| input=128x64 dim=-1   |         5e-05   |          0.00113 |                   22.0703 |
| input=512x256 dim=-1  |         0.00011 |          0.10247 |                  959.571  |
| input=1024x512 dim=-1 |         0.0002  |          0.71466 |                 3613.5    |
| overall               |         0.00011 |          0.10247 |                  959.571  |
## CUDA
### Forward
| inputs                |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-----------------------|-----------------|------------------|---------------------------|
| input=128x64 dim=-1   |         1e-05   |          0.00012 |                  19.631   |
| input=512x256 dim=-1  |         2e-05   |          0.00011 |                   4.49403 |
| input=1024x512 dim=-1 |         0.00013 |          0.00011 |                   0.87675 |
| overall               |         2e-05   |          0.00011 |                   4.49403 |
### Backward
| inputs                |   torch cpu, s. |   tensorgrad, s. |   tensorgrad is slower by |
|-----------------------|-----------------|------------------|---------------------------|
| input=128x64 dim=-1   |         4e-05   |          0.00075 |                  19.6761  |
| input=512x256 dim=-1  |         8e-05   |          0.00087 |                  11.3559  |
| input=1024x512 dim=-1 |         0.00028 |          0.001   |                   3.62592 |
| overall               |         8e-05   |          0.00087 |                  11.3559  |