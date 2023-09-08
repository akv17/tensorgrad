import importlib


class LazyImport:

    def __init__(self, module, err_msg):
        self.module = module
        self.err_msg = err_msg
        self.__module = None

    def __getattr__(self, name):
        self.__import_maybe()
        return getattr(self.__module, name)
    
    def __import_maybe(self):
        if self.__module is None:
            try:
                mod = importlib.import_module(self.module)
                self.__module = mod
            except ImportError as e:
                raise ImportError(self.err_msg) from e
