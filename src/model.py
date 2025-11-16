# src/model.py
"""Model definition and model utilities for MNIST."""
import torch
import torch.nn as nn

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

class SimpleCNN(nn.Module):
    """A small CNN for MNIST classification."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def save_model(model: nn.Module, path: str):
    """Save model weights to disk."""
    torch.save(model.state_dict(), path)

def load_model(path: str, device: str = 'cpu') -> nn.Module:
    """Load model weights from disk and return model on device."""
    model = SimpleCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
