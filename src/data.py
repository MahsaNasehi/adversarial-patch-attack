# src/data.py
"""Data loaders for MNIST with normalization."""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .model import MNIST_MEAN, MNIST_STD

def get_dataloaders(root: str = './data', batch_size: int = 128, num_workers: int = 2):
    """Return train and test dataloaders for MNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train_set = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
