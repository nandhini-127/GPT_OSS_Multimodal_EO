import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import IMAGE_SIZE, BATCH_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR , MODEL_NAME, NUM_CLASSES ,DEVICE , EPOCHS , LEARNING_RATE
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
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def train():
    train_loader, val_loader, _ = get_dataloaders()
    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0

    for epoch in range(EPOCHS):
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

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print(f"Loss: {running_loss/len(train_loader):.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints/vision", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/vision/best_model.pth")
            print("✅ Best model saved!")

    print("Training completed!")

if __name__ == "__main__":
    train()
