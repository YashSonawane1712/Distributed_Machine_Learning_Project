# model/cnn_model.py — CNN architecture for CIFAR-10 classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import NUM_CLASSES, DROPOUT


class FashionCNN(nn.Module):
    """
    CNN for CIFAR-10: 3-channel RGB input (32×32)
    Conv→BN→ReLU→Pool ×3 → FC ×3 → Softmax
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super(FashionCNN, self).__init__()

        # Block 1 — input: 3×32×32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        # After 3 pooling on 32×32: 32→16→8→4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # → 32×16×16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # → 64×8×8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # → 128×4×4

        x = x.view(x.size(0), -1)                        # Flatten → 2048
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def build_model():
    """Factory function — returns a freshly initialized model."""
    return FashionCNN()


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(4, 3, 32, 32)   # CIFAR-10: 3-channel 32×32
    out   = model(dummy)
    print(f"Output shape: {out.shape}")  # (4, 10)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")