import tensorgrad

# define training data and place it on device.
device = 'cpu'
x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
y = [1, 0, 0, 1]
x = tensorgrad.tensor(x).float().to(device)
y = tensorgrad.tensor(y).long().to(device)

# define model, loss function and optimizer and place them on device.
model = tensorgrad.nn.Sequential(
    tensorgrad.nn.Linear(2, 8),
    tensorgrad.nn.ReLU(),
    tensorgrad.nn.Linear(8, 2),
)
model.to(device)
model.train()
optim = tensorgrad.optim.SGD(model.parameters(), lr=0.5)
loss_fn = tensorgrad.nn.CrossEntropyLoss()

# train model for 5k steps.
steps = 5000
for step in range(steps):
    optim.zero_grad()
    outputs = model(x)
    loss = loss_fn(outputs, y)
    loss.backward()
    optim.step()

# run inference on train data.
model.eval()
with tensorgrad.no_grad():
    logits = model(x)
    softmax = logits.softmax(-1).numpy()
    pred = softmax.argmax(-1)
    targets = y.numpy()

# print tragets, prediction and softmax output.
print(f'targets: {targets}')
print(f'prediction: {pred}')
print(f'softmax:')
print(softmax)
