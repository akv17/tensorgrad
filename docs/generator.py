import inspect

import tensorgrad


class DocsGenerator:

    def __init__(self):
        self.buffer = []

    def run(self):
        self._generate_header()
        self._generate_tensor()
        self._generate_modules()
        self._generate_optim()
        self._generate_ctx()
        data = '\n'.join(self.buffer)
        return data

    def _generate_header(self):
        data = HeaderGenerator().run()
        self.buffer.append('# `tensorgrad`')
        self.buffer.append(data)

    def _generate_tensor(self):
        data = TensorGenerator().run()
        self.buffer.append('# `tensorgrad.tensor`')
        self.buffer.append(data)
    
    def _generate_modules(self):
        data = ModulesGenerator(
            header='Collection of common neural network modules',
            module=tensorgrad.nn,
            base_cls=tensorgrad.nn.Module,
        ).run()
        self.buffer.append('# `tensorgrad.nn`')
        self.buffer.append(data)
    
    def _generate_optim(self):
        data = ModulesGenerator(
            header='Collection of optimization algorithms',
            module=tensorgrad.optim,
            base_cls=tensorgrad.optim.Optimizer,
        ).run()
        self.buffer.append('# `tensorgrad.optim`')
        self.buffer.append(data)
    
    def _generate_ctx(self):
        data = CtxGenerator().run()
        self.buffer.append('# `tensorgrad.ctx`')
        self.buffer.append(data)


class HeaderGenerator:
    HEADER = """
`tensorgrad` consists of four main modules:  
- `tensorgrad.tensor`: multi-dimensional array providing core ops and autograd machinery  
- `tensorgrad.nn`: collection of common neural network modules  
- `tensorgrad.optim`: collection of optimization algorithms  
- `tensorgrad.ctx`: utilities for controlling current context  
    """

    def run(self):
        data = _normalize_docstr(self.HEADER)
        return data


class TensorGenerator:

    def __init__(self):
        self.tensor = tensorgrad.Tensor
        self.buffer = []

    def run(self):
        self._generate_doc()
        self._generate_ops()
        data = '\n'.join(self.buffer)
        return data

    def _generate_doc(self):
        doc = _normalize_docstr(self.tensor.__doc__)
        self.buffer.append(doc)
        return doc
    
    def _generate_ops(self):
        ops = self._gather_ops()
        for op in ops:
            self._generate_op(op)

    def _generate_op(self, op):
        try:
            op_method = getattr(self.tensor, op)
            op_doc = op_method.__doc__
            op_doc = _normalize_docstr(op_doc)
            op_sig = inspect.signature(op_method)
            op_params = op_sig._parameters
            op_is_cls_method = 'self' not in op_params
            op_sig = ', '.join([str(v) for k, v in op_params.items() if k != 'self'])
            op_buffer = [f'## `{op}`']
            if op_is_cls_method:
                op_buffer.append(f'*classmethod*  ')
            op_buffer.append(f'{op_doc}  ')
            if op_sig:
                op_buffer.append(f'**Parameters:** `{op_sig}`')
            else:
                op_buffer.append(f'**Parameters:**')
            op_data = '\n'.join(op_buffer)
        except:
            op_data = f'## `{op}`\n{op_doc}  '
        self.buffer.append(op_data)

    def _gather_ops(self):
        ops = []
        for k, v in vars(self.tensor).items():
            if k in (
                '__module__',
                '__doc__',
                '__dict__',
                '__weakref__',
            ):
                continue
            if k.startswith('_') and not k.endswith('__'):
                continue
            if not v.__doc__:
                continue
            ops.append(k)
        ops = sorted(ops)
        return ops


class ModulesGenerator:

    def __init__(self, header, module, base_cls):
        self.header = header
        self.module = module
        self.base_cls = base_cls
        self.buffer = []
    
    def run(self):
        self._generate_header()
        self._generate_modules()
        data = '\n'.join(self.buffer)
        return data
    
    def _generate_header(self):
        self.buffer.append(self.header)
    
    def _generate_modules(self):
        mods = self._gather_modules()
        for mod in mods:
            self._generate_module(mod)
        
    def _generate_module(self, mod):
        doc = mod.__doc__ or ''
        doc = _normalize_docstr(doc)
        buffer = [f'## `{mod.__name__}`', doc]
        data = '\n'.join(buffer)
        self.buffer.append(data)

    def _gather_modules(self):
        mods = []
        for v in vars(self.module).values():
            try:
                if issubclass(v, self.base_cls) and v is not self.base_cls:
                    mods.append(v)
            except TypeError:
                continue
        mods = sorted(mods, key=lambda _m: _m.__name__.split('.')[-1])
        return mods


class CtxGenerator:
    HEADER = """
    Utilities for controlling current context
    """

    def __init__(self):
        self.module = tensorgrad.ctx
        self.buffer = []
    
    def run(self):
        self._generate_header()
        self._generate_funcs()
        data = '\n'.join(self.buffer)
        return data
    
    def _generate_header(self):
        header = _normalize_docstr(self.HEADER)
        self.buffer.append(header)
    
    def _generate_funcs(self):
        funcs = self._gather_funcs()
        for func in funcs:
            self._generate_func(func)
    
    def _generate_func(self, func):
        doc = _normalize_docstr(func.__doc__)
        buffer = [f'## `{func.__name__}`', doc]
        data = '\n'.join(buffer)
        self.buffer.append(data)

    def _gather_funcs(self):
        funcs = [
            v
            for k, v in vars(self.module).items()
            if not k.startswith('_') and v.__doc__
        ]
        return funcs


def _normalize_docstr(v):
    lines = v.split('\n')
    first_line = None
    for i, ln in enumerate(lines):
        if ln.strip():
            first_line = i
            break
    lines = lines[first_line:]
    indent = len(lines[0]) - len(lines[0].lstrip(' '))
    v = '\n'.join(ln[indent:] for ln in lines)
    return v
