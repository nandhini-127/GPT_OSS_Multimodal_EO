# fusion/train_fusion.py

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DEVICE, EPOCHS as DEFAULT_EPOCHS, LEARNING_RATE
from vision.dataset import get_dataloaders
from vision.model import build_model
from fusion.fusion_model import build_fusion_model


# -----------------------------
# Load Vision Model
# -----------------------------
def load_vision_checkpoint(model, checkpoint_path="checkpoints/vision/best_model.pth"):
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✅ Loaded vision weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"❌ Vision checkpoint not found: {checkpoint_path}")


# -----------------------------
# Extract Features
# -----------------------------
def extract_vision_features(model, images):
    with torch.no_grad():
        feats = model.extract_features(images)

        if feats.dim() == 3:
            feats = feats.mean(dim=1)

        elif feats.dim() == 4:
            feats = feats.mean(dim=(2, 3))

    return feats


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(fusion_model, vision_model, loader, class_text_embeddings):
    fusion_model.eval()
    vision_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            vision_feats = extract_vision_features(vision_model, images)
            text_feats = class_text_embeddings[labels]

            outputs = fusion_model(vision_feats, text_feats)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# -----------------------------
# TRAIN FUNCTION (WITH RESUME)
# -----------------------------
def train(epochs, resume=False):
    train_loader, val_loader, _ = get_dataloaders()

    # Vision model
    vision_model = build_model().to(DEVICE)
    load_vision_checkpoint(vision_model)
    vision_model.eval()

    # Load text embeddings
    class_emb_path = "data/processed/class_text_embeddings.pt"
    if not os.path.exists(class_emb_path):
        raise FileNotFoundError("❌ Run llm/embedding_extractor.py first")

    class_text_embeddings = torch.load(class_emb_path, map_location=DEVICE)

    # Detect vision feature dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        dummy_feats = extract_vision_features(vision_model, dummy)
        vision_dim = dummy_feats.shape[-1]

    # Build fusion model
    fusion_model = build_fusion_model(
        vision_dim=vision_dim,
        text_dim=class_text_embeddings.shape[1],
        fusion_dim=512,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(fusion_model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = "checkpoints/fusion_best.pth"

    start_epoch = 1
    best_val_acc = 0

    # -----------------------------
    # RESUME TRAINING
    # -----------------------------
    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)

        fusion_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]

        print(f"🔁 Resuming from epoch {start_epoch}")

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    for epoch in range(start_epoch, epochs + 1):
        fusion_model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            vision_feats = extract_vision_features(vision_model, images)
            text_feats = class_text_embeddings[labels]

            outputs = fusion_model(vision_feats, text_feats)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss = running_loss / total

        val_acc = evaluate(fusion_model, vision_model, val_loader, class_text_embeddings)

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                "epoch": epoch,
                "model_state_dict": fusion_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc
            }, checkpoint_path)

            print(f"✅ Saved best model (Val Acc: {best_val_acc:.2f}%)")

    print(f"\n🎉 Training complete! Best Val Acc: {best_val_acc:.2f}%")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    train(args.epochs, resume=args.resume)