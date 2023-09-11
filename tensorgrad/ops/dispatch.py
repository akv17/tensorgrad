from ..ctx import is_grad_enabled


class OpDispatch:
    _DISPATCH = {}

    @classmethod
    def execute(cls, op, *args, **kwargs):
        inputs = args
        inputs = cls._unpack_inputs_maybe(inputs)
        cls._check_inputs_on_same_device(op=op, inputs=inputs)
        device = inputs[0].device
        requires_grad = any(i.requires_grad for i in inputs)
        op = cls._get_op(op=op, device=device)
        op = op(*args, **kwargs)
        out = op.forward()
        out.requires_grad = requires_grad if is_grad_enabled() else False
        out._children = inputs if is_grad_enabled() else ()
        out._op = op
        return out

    @classmethod
    def register(cls, op, device):
        def _deco(impl):
            key = (op, device)
            if key in cls._DISPATCH:
                msg = f'op already registered: {key} -> {cls._DISPATCH[key]}'
                raise KeyError(msg)
            impl.NAME = op
            cls._DISPATCH[key] = impl
            return impl
        return _deco

    @classmethod
    def _get_op(cls, op, device):
        key = (op, device)
        if key not in cls._DISPATCH:
            msg = f'unknown op: {key}'
            raise KeyError(msg)
        op = cls._DISPATCH[key]
        return op

    @classmethod
    def _unpack_inputs_maybe(cls, inputs):
        # not using any fancy iterators not to record select ops when indexing into inputs.
        # may also be solved via running in no_grad mode.
        accum = []
        for i in inputs:
            i = [i] if not isinstance(i, (list, tuple)) else i
            accum.extend(i)
        return accum

    @classmethod
    def _check_inputs_on_same_device(cls, op, inputs):
        device = None
        for i, inp in enumerate(inputs):
            if device is None:
                device = inp.device
            if inp.device != device:
                msg = f'Expected all inputs of {op} on the same device but got ({device, inp.device})'
                raise Exception(msg)
