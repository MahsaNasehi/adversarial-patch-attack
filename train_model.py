# src/train_model.py
"""Script to train a MNIST classifier or load existing weights."""
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from src.model import SimpleCNN, save_model
from src.data import get_dataloaders

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        running = 0.0
        for x,y in tqdm(train_loader, desc=f"Train epoch {ep+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        val_acc = evaluate(model, test_loader, device)
        print(f"Epoch {ep+1} loss: {running/len(train_loader):.4f} val_acc: {val_acc*100:.2f}%")
    save_model(model, args.save_path)
    print('Model saved to', args.save_path)

def evaluate(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_path', type=str, default='mnist_cnn.pth')
    args = parser.parse_args()
    train(args)
