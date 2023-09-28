import tensorgrad

# define data and compute linear function.
x = tensorgrad.tensor.rand(2, 4, requires_grad=True)
w = tensorgrad.tensor.rand(3, 4, requires_grad=True)
b = tensorgrad.tensor.rand(3, requires_grad=True)
f = x.matmul(w.transpose(1, 0)) + b
# dont destroy the graph to be able to visualize it.
f.backward(destroy_graph=False)
# render the graph as PIL image.
x.name = 'x'
w.name = 'w'
b.name = 'b'
f.name = 'output'
image = f.render()
image.show()
