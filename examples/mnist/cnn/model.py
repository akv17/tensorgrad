import tensorgrad


class Block(tensorgrad.nn.Module):

    def __init__(self, in_channels, out_channels, padding='same'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = tensorgrad.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            padding=padding
        )
        self.bn = tensorgrad.nn.BatchNorm2d(self.out_channels)
        self.pool = tensorgrad.nn.AvgPool2d((2, 2))
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x.relu()
        x = self.pool(x)
        return x


class CNN(tensorgrad.nn.Module):

    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone = tensorgrad.nn.Sequential(
            Block(in_channels=self.in_channels, out_channels=8),
            Block(in_channels=8, out_channels=16, padding=2),
            Block(in_channels=16, out_channels=32),
        )
        self.dropout = tensorgrad.nn.Dropout(0.1)
        self.flatten = tensorgrad.nn.Flatten()
        self.head = tensorgrad.nn.Linear(4 * 4 * 32, self.num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.head(x)
        return x
