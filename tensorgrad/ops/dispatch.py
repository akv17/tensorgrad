class OpDispatch:
    _DISPATCH = {}

    @classmethod
    def execute(cls, op, *args, **kwargs):
        device = args[0].device
        key = (op, device)
        op = cls._DISPATCH[key]
        op = op(*args, **kwargs)
        out = op.forward()
        out.requires_grad = any(a.requires_grad for a in args)
        out._children = args
        out._op = op
        return out

    @classmethod
    def register(cls, op, device):
        def _deco(impl):
            key = (op, device)
            if key in cls._DISPATCH:
                msg = f'op already registered: {key} -> {cls._DISPATCH[key]}'
                raise KeyError(msg)
            cls._DISPATCH[key] = impl
            return impl
        return _deco
