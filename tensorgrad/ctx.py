_IS_GRAD_ENABLED = True


def is_grad_enabled():
    return _IS_GRAD_ENABLED


class no_grad:

    def __enter__(self):
        global _IS_GRAD_ENABLED
        _IS_GRAD_ENABLED = False
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _IS_GRAD_ENABLED
        _IS_GRAD_ENABLED = True
        return False
