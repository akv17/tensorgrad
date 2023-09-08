def accumulate_broadcasted_grad(np, x, g):
    """accumulates grad `g` of `x` over broadcasted dims of `x`"""
    g = g.copy()
    x_shape = x.shape
    g_shape = g.shape
    if len(x_shape) > len(g_shape):
        return g
    # right align dims following numpy broadcasting rules.
    diff = len(g_shape) - len(x_shape)
    x_dims = [None] * diff + list(range(len(x_shape)))
    g_dims = list(range(len(g_shape)))
    dims_zip = [*zip(x_dims, g_dims)]
    for xd, gd in reversed(dims_zip):
        # expanded dim.
        if xd is None:
            g = g.sum(gd)
            continue
        xs = x_shape[xd]
        gs = g_shape[gd]
        # dim of size one which was actually broadcasted.
        # we need to keep it so expanding it back after reduce.
        # this is essentially `keepdim=True` behavior.
        if xs == 1 and gs != 1:  
            g = g.sum(xd)
            g = np.expand_dims(g, xd)
    return g
