# `tensorgrad`
`tensorgrad` consists of four main modules:  
- `tensorgrad.tensor`: multi-dimensional array providing core ops and autograd machinery  
- `tensorgrad.nn`: collection of common neural network modules  
- `tensorgrad.optim`: collection of optimization algorithms  
- `tensorgrad.ctx`: utilities for controlling current context  
    
# `tensorgrad.tensor`
Implementation of a multi-dimensional array of specific data type on a specific device.  
Provides core arithmetic, indexing and shape operations, factory functions and utilities.  

Arithmetic and shape operations preserve data type and device placement.  
Factory functions accept desired data type and device as arguments.  
Utilities mainly provide means to cast data types or move data to another device.  

Supported data types:  
- `tensorgrad.DTYPE.FLOAT32`  
- `tensorgrad.DTYPE.FLOAT64`  
- `tensorgrad.DTYPE.INT32`  
- `tensorgrad.DTYPE.INT64`  
- `tensorgrad.DTYPE.BOOL`  

Supported devices:  
- `tensorgrad.DEVICE.CPU`  
- `tensorgrad.DEVICE.CUDA`  

All arithmetic, indexing and shape operations are automatically recorded into computational graph by autograd engine.  
Each node in the graph is a tensor being result of some operation.  
Such graph may then be traversed forward to compute output or backward to perform reverse-mode differentiation.  
Graph traversal may be started from any node.  
Tensor may be excluded from the graph by setting `requires_grad` to `False` or calling `detach()`.  

## `__add__`
Add tensor to other tensor or constant element-wise  
**Parameters:** `other`
## `__getitem__`
Retreive tensor slice by integer index or by a mask  
**Parameters:** `slice_`
## `__init__`
Construct tensor explicitly from data with given spec  
**Parameters:** `data, dtype=None, device=None, requires_grad=True`
## `__invert__`
Invert boolean tensor  
**Parameters:**
## `__mul__`
Multiply tensor with other tensor or constant element-wise  
**Parameters:** `other`
## `__neg__`
Negate tensor (multiply by -1.0)  
**Parameters:**
## `__pow__`
Raise tensor to the power  
**Parameters:** `value`
## `__radd__`
Add tensor to other tensor or constant element-wise  
**Parameters:** `other`
## `__rmul__`
Multiply tensor with other tensor or constant element-wise  
**Parameters:** `other`
## `__sub__`
Subtract other tensor or constant from tensor element-wise  
**Parameters:** `other`
## `__truediv__`
Divide tensor by other tensor or constant element-wise  
**Parameters:** `other`
## `arange`
*classmethod*  
Construct range tensor with given spec  
**Parameters:** `n, dtype=None, device=None, requires_grad=True`
## `avg_pool2d`
Perform 2D average pooling on tensor with given kernel and parameters  
**Parameters:** `kernel_size, stride=None, padding=None`
## `backward`
Run autograd backward pass starting from tensor as a leaf node  
**Parameters:** `upstream=None`
## `bernoulli`
*classmethod*  
Construct tensor from random bernoulli distribution with given spec and parameter `p`  
**Parameters:** `p, shape, dtype=None, device=None, requires_grad=True`
## `bool`
Cast tensor to bool  
**Parameters:** `inplace=False`
## `cat`
Concate tensor with other tensors along dimension  
**Parameters:** `tensors, dim`
## `concat`
Concate tensor with other tensors along dimension  
**Parameters:** `tensors, dim`
## `conv2d`
Perform 2D convolution on tensor with given kernel and parameters  
**Parameters:** `kernel, bias=None, stride=None, padding=None`
## `copy`
Copy of the tensor  
**Parameters:**
## `cpu`
Move tensor to CPU device  
**Parameters:** `inplace=True`
## `cuda`
Move tensor to CUDA device  
**Parameters:** `inplace=True`
## `detach`
Detach tensor from autograd tracking  
**Parameters:**
## `device`
## `dtype`
## `empty`
*classmethod*  
Construct empty tensor with given spec  
**Parameters:** `*shape, dtype=None, device=None, requires_grad=True`
## `exp`
Compute exponent element-wise  
**Parameters:**
## `float`
Cast tensor to float32  
**Parameters:** `inplace=False`
## `from_data`
Construct new tensor with given data  
**Parameters:** `data`
## `item`
Convert tensor to native python list or scalar  
**Parameters:**
## `log`
Compute natural logarithm element-wise  
**Parameters:**
## `long`
Cast tensor to int64  
**Parameters:** `inplace=False`
## `lookup`
Lookup indices into tensor provided by mask  
**Parameters:** `mask`
## `masked_fill_`
Fill tensor with value at positions where mask is `True`. Mask must be of the same shape as tensor.  
**Parameters:** `mask, value`
## `matmul`
Perform tensor multiplication with other tensor  
**Parameters:** `other`
## `max`
Reduce tensor by computing max over dimension or globally  
**Parameters:** `dim=None`
## `max_pool2d`
Perform 2D max pooling on tensor with given kernel and parameters  
**Parameters:** `kernel_size, stride=None, padding=None`
## `mean`
Reduce tensor by computing mean over dimension or globally. Supports reduce over multiple dimensions.  
**Parameters:** `dim=None, keepdim=False`
## `min`
Reduce tensor by computing min over dimension or globally  
**Parameters:** `dim=None`
## `ndim`
## `numel`
Total number of elements in the tensor  
**Parameters:**
## `numpy`
Convert tensor to numpy array preserving data type  
**Parameters:**
## `ones`
*classmethod*  
Construct tensor of ones with given spec  
**Parameters:** `*shape, dtype=None, device=None, requires_grad=True`
## `ones_like`
Construct new tensor of ones with the same spec as tensor  
**Parameters:**
## `permute`
Permute tensor dimensions  
**Parameters:** `*dims`
## `rand`
*classmethod*  
Construct tensor from random uniform distribution [0, 1) with given spec  
**Parameters:** `*shape, dtype=None, device=None, requires_grad=True`
## `randint`
*classmethod*  
Construct tensor of random integers in range [low, high) with given spec  
**Parameters:** `low, high, shape, dtype=None, device=None, requires_grad=True`
## `randn`
*classmethod*  
Construct tensor from random normal distribution with given spec and mu and sigma  
**Parameters:** `*shape, mu=0.0, sigma=1.0, dtype=None, device=None, requires_grad=True`
## `relu`
Compute ReLU function element-wise  
**Parameters:**
## `render`
Render autograd graph from tensor as a leaf node  
**Parameters:**
## `reshape`
Reshape tensor to a new shape  
**Parameters:** `*shape`
## `shape`
## `sigmoid`
Compute sigmoid function element-wise  
**Parameters:**
## `softmax`
Compute softmax function over 1D slice along given dimension  
**Parameters:** `dim`
## `sqrt`
Compute square root element-wise  
**Parameters:**
## `squeeze`
Drop tensor dimension of size one  
**Parameters:** `dim`
## `sum`
Reduce tensor by computing sum over dimension or globally. Supports reduce over multiple dimensions.  
**Parameters:** `dim=None, keepdim=False`
## `to`
Move tensor to device  
**Parameters:** `device, inplace=True`
## `tolist`
Convert tensor to native python list or scalar  
**Parameters:**
## `transpose`
Permute tensor dimensions  
**Parameters:** `*dims`
## `unsqueeze`
Add new dimension of size one  
**Parameters:** `dim`
## `zeros`
*classmethod*  
Construct tensor of zeros with given spec  
**Parameters:** `*shape, dtype=None, device=None, requires_grad=True`
## `zeros_like`
Construct new tensor of zeros with the same spec as tensor  
**Parameters:**
# `tensorgrad.nn`
Collection of common neural network modules
## `AvgPool2d`
Performs average pooling over spatial dimensions of a batch of 3D tensors.  
Expects tensors in channel-first format.  

**Parameters:**  
- `kernel_size: tuple, int:` kernel size  
- `stride: tuple, int: None:` stride of sliding window  
- `padding: tuple, int: 0:` size of input padding along both spatial dimensions  

**Input:** `(B, C, H_in, W_in)`  
**Output:** `(B, C, H_out, W_out)`  
**Weights:** `None`  

## `BatchNorm1d`
Applies Batch Normalization to 2D inputs.  
Tracks running estimates of mean and variance to use in inference.  

**Parameters:**  
- `num_features: int:` number of features in input  
- `eps: float: 1e-5:` small value for numerical stability  
- `momentum: float: 0.1:` proportion of new observation when accumulating running estimates  
- `track_running_stats: bool: True:` tracks running estimates of mean and variance if set to `True`  

**Input:** `(B, F)`  
**Output:** `(B, F)`  
**Weights:**  
- `weight: (F,):` learnable gamma  
- `bias: (F,):` learnable beta  

## `BatchNorm2d`
Applies Batch Normalization to a batch of 3D tensors.  
Expects tensors in channel-fitst format.  
Tracks running estimates of mean and variance to use in inference.  

**Parameters:**  
- `num_features: int:` number of channels in each 3D input  
- `eps: float: 1e-5:` small value for numerical stability  
- `momentum: float: 0.1:` proportion of new observation when accumulating running estimates  
- `track_running_stats: bool: True:` tracks running estimates of mean and variance if set to `True`  

**Input:** `(B, C, H, W)`  
**Output:** `(B, C, H, W)`  
**Weights:**  
- `weight: (C,):` learnable gamma  
- `bias: (C,):` learnable beta  

## `Conv2d`
Performs 2D convolution over spatial dimensions of a batch of 3D tensors.  
Expects tensors in channel-first format.  

**Parameters:**  
- `in_channels: int:` number of channels in input  
- `out_channels: int:` number of channels in output  
- `kernel_size: tuple, int:` size of the convolution kernel  
- `stride: tuple, int: 1:` stride of the sliding window  
- `padding: tuple, int, same: 0:` size of the input padding along both spatial dimensions. `same` will preserve the same size as input  
- `bias: bool:` will add bias if set to `True`  

**Input:** `(B, C, H_in, W_in)`  
**Output:** `(B, C, H_out, W_out)`  
**Weights:**  
- `weight: (C_out, C_in, H_kernel, W_kernel):` learnable weights of the convolution kernel
- `bias: (C_out,):` learnable bias if `bias` was set to `True`  

## `CrossEntropyLoss`
Computes cross-entropy loss function between input logits and targets.  
Input logits are expected to be unnormalized as softmax will be applied internally.  

**Parameters**:  
- `reduction: str: mean, sum:` type of reduction to final scalar value  

**Input:**  
- `outputs: (B, num_classes):` input logits for each sample  
- `targets: (B,):` indices as integers of a target class for each sample  

**Output:** `scalar`  
**Weights:** `None`  

## `Dropout`
Applies dropout to input.  
Basically, zeroes out each scalar element of the input with probability `p`.  
Operates only in training mode.  

**Parameters:**  
- `p: int:` probability of element zeroing  

**Input:** `(*)`  
**Output:** `(*)`  
**Weights:** `None`  

## `Embedding`
Performs embedding table lookup.  
Expects inputs to contain indices into the embedding table.  

**Parameters:**
- `num_embeddings: int:` number of embeddings (aka number of rows in the table)
- `embedding_dim: int:` dimensionality of each embedding (aka number of columns in the table)

**Input:** `(*)`  
**Output:** `(*, E)` attaches new dimension with embeddings  
**Weights:**  
- `weight: (num_embeddings, embedding_dim):` learnable embedding table

## `Flatten`
Flattens batch of input tensors to a 2D tensor preserving batch size.  
Basically, flattens all dimensions except the first one.  

**Parameters:** `None`  
**Input:** `(B, *)`  
**Output:** `(B, F)`  
**Weights:** `None`  

## `Identity`
Applies indentity transform to input.  
Basically does nothing, a no-op.  

**Parameters:** `None`  
**Input:** `(*)`  
**Output:** `(*)`  
**Weights:** `None`  

## `LayerNorm`
Applies Layer Normalization to input.  

**Parameters:**  
- `normalized_shape: int, tuple:` shape to normalize over as a contiguous subset of input shape starting from the end  
- `eps: float: 1e-5:` small value for numerical stability  

**Input:** `(*)`  
**Output:** `(*)`  
**Weights:**  
- `weight: normalized_shape:` learnable gamma  
- `bias: normalized_shape:` learnable beta  

## `Linear`
Applies linear transformation `x * w.T + b` to an input.  
Input may have any number of dimenstions but no less than 2.  
Size of the last dimension must be equal to `in_features`.  

**Parameters:**  
- `in_features: int:` size of each input element  
- `out_features: int:` size of each output element  
- `bias: bool:` will add bias if set to `True`  

**Input:** `(*, in_features)`  
**Output:** `(*, out_features)`  
**Weights:**  
- `weight: (out_features, in_features):` learnable weights  
- `bias: (out_features,):` learnable bias if `bias` was set to `True`  

## `MSELoss`
Computes mean-squared-error loss function between inputs and targets.  
Inputs and targets are expected to have the same shape and dtype.  

**Parameters**:  
- `reduction: str: mean, sum:` type of reduction to final scalar value  

**Input:**  
- `outputs: (*)`  
- `targets: (*)`  

**Output:** `scalar`  
**Weights:** `None`  

## `MaxPool2d`
Performs max pooling over spatial dimensions of a batch of 3D tensors.  
Expects tensors in channel-first format.  

**Parameters:**  
- `kernel_size: tuple, int:` kernel size  
- `stride: tuple, int: None:` stride of sliding window  
- `padding: tuple, int: 0:` size of input padding along both spatial dimensions  

**Input:** `(B, C, H_in, W_in)`  
**Output:** `(B, C, H_out, W_out)`  
**Weights:** `None`  

## `MultiheadAttention`
Performs multihead attention as described in *Attention Is All You Need*.  
Operates only on 3D tensors of shape `(batch_size, seq_len, embed_dim)`.  
It's also possible to pass a mask preventing attention to specific positions.  

**Parameters:**  
- `embed_dim: int:` embedding dimensionality (aka d_model)  
- `num_heads: int:` number of attention heads  

**Input:**  
- `q: (B, T_q, E)`: query tensor  
- `k: (B, T_k, E)`: key tensor  
- `v: (B, T_k, E)`: value tensor  
- `attn_mask (optional): bool: (B, T_q, T_k) or (T_q, T_k)`: boolean mask preventing attention to specific positions. position is not attended if value in the mask is `True`.  

**Output:** `(B, T_q, E)`  
**Weights:**  
- `q_weight: (embed_dim, embed_dim):` learnable weights of query projection  
- `k_weight: (embed_dim, embed_dim):` learnable weights of key projection  
- `v_weight: (embed_dim, embed_dim):` learnable weights of value projection  
- `o_weight: (embed_dim, embed_dim):` learnable weights of merged output projection  

## `ReLU`
Applies ReLU function element-wise.  

**Parameters:** `None`  
**Input:** `(*)`  
**Output:** `(*)`  
**Weights:** `None`  

## `Sequential`
Applies a sequence of modules in order one after another.  

**Parameters**:  
- `modules: Iterable[Module]:` iterable of modules to apply  

**Input:** `(*)`  
**Output:** `(*)`  
**Weights:** `None`  

## `Sigmoid`
Applies sigmoid function element-wise.  

**Parameters:** `None`  
**Input:** `(*)`  
**Output:** `(*)`  
**Weights:** `None`  

## `Softmax`
Applies softmax function to a 1D slice of input along given dimension.  

**Parameters:**  
- `dim: int:` dimension along which softmax is applied  

**Input:** `(*)`  
**Output:** `(*)`  
**Weights:** `None`  

# `tensorgrad.optim`
Collection of optimization algorithms
## `Adam`
Implements Adam algorithm with original defaults.  

**Parameters:**  
- `parameters: Iterable[Parameter]:` parameters to optimize  
- `lr: float: 1e-3:` learning rate  
- `betas: tuple: (0.9, 0.999):` first and second moments' scales  
- `eps: float: 1e-8:` small value for numerical stability  

## `SGD`
Implements stochastic gradient descent with momentum.  

**Parameters:**  
- `parameters: Iterable[Parameter]:` parameters to optimize  
- `lr: float:` learning rate  
- `momentum: float: None:` momentum factor  

# `tensorgrad.ctx`
Utilities for controlling current context

## `is_grad_enabled`
Return whether gradient tracking is currently enabled
## `is_cuda_available`
Return whether CUDA device is available
## `no_grad`
Context manager disabling gradient tracking