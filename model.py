import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)  # 1x28x28 -> 32x26x26 , RF: 3x3
        x = F.max_pool2d(
            F.relu(self.conv2(x), 2), 2
        )  # 32x26x26 -> 64x24x24 -> 64x13x13 , RF: 6x6
        x = F.relu(self.conv3(x), 2)  # 64x13x13 -> 128x11x11, RF: 10x10
        x = F.max_pool2d(
            F.relu(self.conv4(x), 2), 2
        )  # 128x11x11 -> 256x9x9 -> 256x4x4, RF: 16x16
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
