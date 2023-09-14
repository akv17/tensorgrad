_IS_GRAD_ENABLED = True


def is_grad_enabled():
    """Return whether gradient tracking is currently enabled"""
    return _IS_GRAD_ENABLED


def is_cuda_available():
    """Return whether CUDA device is available"""
    return True


class no_grad:
    """Context manager disabling gradient tracking"""

    def __enter__(self):
        global _IS_GRAD_ENABLED
        _IS_GRAD_ENABLED = False
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _IS_GRAD_ENABLED
        _IS_GRAD_ENABLED = True
        return False
