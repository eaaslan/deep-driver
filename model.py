import torch
import torch.nn as nn

class DrowsinessDetectorCNN(nn.Module):
    def __init__(self):
        super(DrowsinessDetectorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Since the shape before fc1 is 32x65536, adjust the Linear layer accordingly
        self.fc1 = nn.Linear(65536, 256)  # Correcting to match the actual flattening size
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Ensure flattening matches the tensor shape
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
