# tensorgrad
TODO:
- [x] masking
- [x] relu
- [x] sigmoid
- [x] softmax
- [x] matmul
- [x] linear
- [x] cross-entropy
- [x] sgd
- [x] adam

- [x] maxreduce
- [x] conv2d
- [x] batchnorm2d
- [x] maxpool2d

- [] multiattention
- [] layernorm

[] arg checks
    [] OpDispatch
    [] custom checks for each op

PROBLEMS:
- how to control requires_grad for outputs of ops
- numpy export for cpu ops is lame atm
- save stuff for backward during forward
- support multidim reductions (this will also speedup batchnorm2d)

TORCH API NOT IMPLEMENTED YET:
- Parameter type
- auto detection of Module params
- auto naming of Module params
- no_grad
- train / eval
- save
- load
