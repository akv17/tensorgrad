import tempfile


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
        visited.add(id(node))
        op = node._op
        op_key = f'_op_{id(node)}'
        dot.node(str(id(node)), f'{node.name}\n\nshape={list(node.shape)}\n\ndata={node.data.data}\n\ngrad={node.grad.data}', shape='box')
        if op is not None:
            dot.node(op_key, op)
        for ch in node._children:
            # dot.node(str(id(ch)), f'{ch.name}\nshape={list(ch.shape)}\ndata={node.data.data}\n\ngrad={node.grad.data}', shape='box')
            nodes.append(ch)
            if op is not None:
                dot.edge(str(id(ch)), op_key)
        if op is not None:
            dot.edge(op_key, str(id(node)))
    
    with tempfile.TemporaryDirectory() as tmp:
        dot.render(filename='g', directory=tmp)
        fp = f'{tmp}/g.png'
        im = Image.open(fp)
    im.show()
