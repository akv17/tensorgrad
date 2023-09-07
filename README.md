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

- [x] multiattention
- [x] layernorm
- [x] embedding
- [x] dropout

- [] arg checks
    - [] OpDispatch
    - [] custom checks for each op

PROBLEMS:
- how to control requires_grad for outputs of ops
- [x] numpy export for cpu ops is lame atm
- [irrelevant] save stuff for backward during forward
- [x] support multidim reductions (this will also speedup batchnorm2d)

TORCH API NOT IMPLEMENTED YET:
- [x] Parameter type
- [x] auto detection of Module params
- [x] auto naming of Module params
- [x] no_grad
- [] train / eval
- [] save
- [] load
