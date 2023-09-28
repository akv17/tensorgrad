import tempfile
from uuid import uuid4


def render_graph(node):
    import graphviz
    from PIL import Image
    dot = graphviz.Digraph()
    dot.format = 'png'
    
    nodes = [node]
    visited = set()
    while nodes:
        node = nodes.pop(0)
        if id(node) in visited:
            continue
        node_name = node.name or f'tensor@{str(uuid4())[:8]}'
        visited.add(id(node))
        op = node._op
        op_key = f'_op_{id(node)}'
        node_data = _normalize_data_repr(node.data)
        node_grad = _normalize_data_repr(node.grad)
        dot.node(str(id(node)), f'{node_name}\n\nshape={list(node.shape)}\n\ndata={node_data}\n\ngrad={node_grad}', shape='box')
        if op is not None:
            dot.node(op_key, str(op.NAME.value))
        for ch in node._children:
            nodes.append(ch)
            if op is not None:
                dot.edge(str(id(ch)), op_key)
        if op is not None:
            dot.edge(op_key, str(id(node)))
    
    with tempfile.TemporaryDirectory() as tmp:
        dot.render(filename='g', directory=tmp)
        fp = f'{tmp}/g.png'
        im = Image.open(fp)
    return im

def _normalize_data_repr(data, max_size=120):
    data = str(data)
    if len(data) > max_size:
        size = max_size // 2
        data = data[:size] + ' ... ' + data[-size:]
    return data
