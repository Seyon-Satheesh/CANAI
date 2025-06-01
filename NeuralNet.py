import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)

        self.pool = nn.MaxPool2d(2, stride=2)

        # Calculate the size of the flattened layer dynamically
        # We can do this by passing a dummy tensor through the conv and pool layers
        dummy_input = torch.randn(1, 3, 128, 128) # Batch size of 1, 3 channels, 128x128 image
        x = F.relu(self.conv1(dummy_input))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        flattened_size = x.size(1) * x.size(2) * x.size(3) # Calculate the flattened size

        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x