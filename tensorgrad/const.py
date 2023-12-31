from enum import Enum


class DTYPE(str, Enum):
    FLOAT32 = 'float32'
    FLOAT64 = 'float64'
    INT32 = 'int32'
    INT64 = 'int64'
    BOOL = 'bool'


class DEVICE(str, Enum):
    CPU = 'cpu'
    CUDA = 'cuda'


class OP(str, Enum):
    ADD = 'add'
    SUB = 'sub'
    MUL = 'mul'
    DIV = 'div'
    INVERT = 'invert'
    
    POW = 'pow'
    EXP = 'exp'
    LOG = 'log'
    MASKED_FILL = 'masked_fill_'
    
    SUM_REDUCE = 'sum_reduce'
    MEAN_REDUCE = 'mean_reduce'
    MAX_REDUCE = 'max_reduce'
    MIN_REDUCE = 'min_reduce'

    SQUEEZE = 'squeeze'
    UNSQUEEZE = 'unsqueeze'
    RESHAPE = 'reshape'
    PERMUTE = 'permute'
    SELECT = 'select'
    LOOKUP = 'lookup'
    CONCAT = 'concat'

    RELU = 'relu'
    SIGMOID = 'sigmoid'
    SOFTMAX = 'softmax'
    MATMUL = 'matmul'
    CONV2D = 'conv2d'
    MAX_POOL2D = 'max_pool2d'
    AVG_POOL2D = 'avg_pool2d'
