def accumulate_broadcasted_grad(x, g):
    """accumulates grad `g` of `x` over broadcast-inserted dims of `x` if `x` was broadcasted"""
    g = g.copy()
    x_shape = x.shape
    g_shape = g.shape
    if len(x_shape) >= len(g_shape):
        return g
    dim_diff = len(g_shape) - len(x_shape)
    x_broad_dims = list(range(dim_diff))
    while x_broad_dims:
        dim = x_broad_dims.pop(-1)
        g = g.sum(dim=dim)
    return g
