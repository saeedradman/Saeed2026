import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Custom CNN built from scratch for MRI image classification.
    Designed for small-to-medium medical datasets.
    """
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional Blocks 
        # Block 1: low-level features (edges, contrast)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: mid-level patterns
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: high-level semantic features
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Max pooling reduces spatial size and computation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pooling allows flexible input size
        # Output will always be (128, 7, 7)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 2)  # Binary classification (MS / Control)

    def forward(self, x):
        """
        Forward propagation of the network.
        """

        # Convolutional block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Convolutional block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Convolutional block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Adaptive pooling for size consistency
        x = self.adaptive_pool(x)

        # Flatten feature maps into vector
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x