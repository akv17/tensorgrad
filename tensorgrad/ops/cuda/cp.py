class _CupyProxy:

    def __init__(self):
        self.__cupy = None

    def __getattr__(self, name):
        self.__import_maybe()
        return getattr(self.__cupy, name)
    
    def __import_maybe(self):
        try:
            import cupy
            self.__cupy = cupy
        except ImportError:
            raise Exception('cuda not available')


class CupyProvider:
    np = _CupyProxy()
