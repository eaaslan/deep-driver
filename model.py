import torch
import torch.nn as nn

class DrowsinessDetectorCNN(nn.Module):
    def __init__(self):
        super(DrowsinessDetectorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)  # Output size stays nearly the same
        self.pool = nn.MaxPool2d(2, 2)  # Size is halved
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # Output size stays nearly the same
        self.pool2 = nn.MaxPool2d(2, 2)  # Size is halved again
        # Calculate correct dimensions:
        # After pooling twice, size is quartered in both dimensions (assuming no padding issues)
        final_dim = (227 // 4) * (227 // 4) * 64  # Final dimension before the fully connected layer
        self.fc1 = nn.Linear(final_dim, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


