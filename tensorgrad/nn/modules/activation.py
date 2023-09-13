from .base import Module


class Identity(Module):
    """
    Applies indentity transform to input.  
    Basically does nothing, a no-op.  

    **Parameters:** `None`  
    **Input:** `(*)`  
    **Output:** `(*)`  
    **Weights:** `None`  
    """

    def forward(self, x):
        return x


class ReLU(Module):
    """
    Applies ReLU function element-wise.  

    **Parameters:** `None`  
    **Input:** `(*)`  
    **Output:** `(*)`  
    **Weights:** `None`  
    """
    
    def forward(self, x):
        x = x.relu()
        return x


class Sigmoid(Module):
    """
    Applies sigmoid function element-wise.  

    **Parameters:** `None`  
    **Input:** `(*)`  
    **Output:** `(*)`  
    **Weights:** `None`  
    """

    def forward(self, x):
        x = x.sigmoid()
        return x


class Softmax(Module):
    """
    Applies softmax function to a 1D slice of input along given dimension.  

    **Parameters:**  
    - `dim: int:` dimension along which softmax is applied  

    **Input:** `(*)`  
    **Output:** `(*)`  
    **Weights:** `None`  
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.softmax(dim=self.dim)
        return x
