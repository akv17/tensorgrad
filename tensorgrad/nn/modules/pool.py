from .base import Module


class _GeneralizedPool2d(Module):
    _OP = None
    
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._op = f'{self._OP.lower()}_pool2d'

        self._normalize_kernel_size()
        self._normalize_stride()
        self._normalize_padding()

    def forward(self, x):
        op = getattr(x, self._op)
        out = op(kernel_size=self._kernel_size, stride=self._stride, padding=self._padding)
        return out

    def _normalize_kernel_size(self):
        value = self.kernel_size
        self._kernel_size = (value, value) if isinstance(value, int) else value
    
    def _normalize_stride(self):
        value = self.stride
        if value is None:
            self._stride = self._kernel_size
        else:
            self._stride = (value, value) if isinstance(value, int) else value
    
    def _normalize_padding(self):
        value = self.padding
        self._padding = (value, value) if isinstance(value, int) else value


class MaxPool2d(_GeneralizedPool2d):
    """
    Performs max pooling over spatial dimensions of a batch of 3D tensors.  
    Expects tensors in channel-first format.  

    **Parameters:**  
    - `kernel_size: tuple, int:` kernel size  
    - `stride: tuple, int: None:` stride of sliding window  
    - `padding: tuple, int: 0:` size of input padding along both spatial dimensions  

    **Input:** `(B, C, H_in, W_in)`  
    **Output:** `(B, C, H_out, W_out)`  
    **Weights:** `None`  
    """

    _OP = 'max'


class AvgPool2d(_GeneralizedPool2d):
    """
    Performs average pooling over spatial dimensions of a batch of 3D tensors.  
    Expects tensors in channel-first format.  

    **Parameters:**  
    - `kernel_size: tuple, int:` kernel size  
    - `stride: tuple, int: None:` stride of sliding window  
    - `padding: tuple, int: 0:` size of input padding along both spatial dimensions  

    **Input:** `(B, C, H_in, W_in)`  
    **Output:** `(B, C, H_out, W_out)`  
    **Weights:** `None`  
    """

    _OP = 'avg'
