# vision/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DEVICE, EPOCHS, LEARNING_RATE
from vision.dataset import get_dataloaders
from vision.model import build_model


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total


def train():
    train_loader, val_loader, _ = get_dataloaders()

    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ✅ Checkpoint path
    checkpoint_path = "checkpoints/vision/checkpoint.pth"
    os.makedirs("checkpoints/vision", exist_ok=True)

    start_epoch = 0
    best_acc = 0

    # ✅ LOAD CHECKPOINT (if exists)
    if os.path.exists(checkpoint_path):
        print("🔄 Loading checkpoint...")

        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

        print(f"✅ Resuming from epoch {start_epoch}")

    # ✅ TRAIN LOOP
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_acc = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Loss: {running_loss/len(train_loader):.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")

        # ✅ SAVE CHECKPOINT EVERY EPOCH
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, checkpoint_path)

        # ✅ SAVE BEST MODEL
        if val_acc > best_acc:
            best_acc = val_acc

            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": val_acc
            }, "checkpoints/vision/best_model.pth")

            print("✅ Best model saved!")

    print("🎉 Training complete!")


if __name__ == "__main__":
    train()