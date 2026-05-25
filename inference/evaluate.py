import json
import os
import sys

import torch
from torch.utils.data import DataLoader

# Ensure repo root is on sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DEVICE
from vision.dataset import get_dataloaders
from vision.model import build_model
from fusion.fusion_model import build_fusion_model


def load_vision_model(checkpoint_path="checkpoints/vision_best.pth"):
    model = build_model().to(DEVICE)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        # Vision checkpoint is a timm state_dict; our VisionModel wraps it as .model
        model.model.load_state_dict(state)
        print(f"Loaded vision weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Vision checkpoint not found: {checkpoint_path}")
    model.eval()
    return model


def load_fusion_model(vision_dim: int, text_dim: int, checkpoint_path="checkpoints/fusion_best.pth"):
    model = build_fusion_model(vision_dim=vision_dim, text_dim=text_dim, fusion_dim=512).to(DEVICE)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
        print(f"Loaded fusion weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Fusion checkpoint not found: {checkpoint_path}")
    model.eval()
    return model


def evaluate():
    _, _, test_loader = get_dataloaders()

    vision_model = load_vision_model()

    # Determine vision feature dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        vision_feats = vision_model.extract_features(dummy)
        if vision_feats.dim() == 4:
            vision_feats = vision_feats.mean(dim=(2, 3))
        vision_dim = vision_feats.shape[-1]

    # Load class embeddings
    class_emb_path = "data/processed/class_text_embeddings.pt"
    if not os.path.exists(class_emb_path):
        raise FileNotFoundError(f"Class text embeddings not found: {class_emb_path}")
    class_text_embeddings = torch.load(class_emb_path, map_location=DEVICE, weights_only=True)

    fusion_model = load_fusion_model(vision_dim=vision_dim, text_dim=class_text_embeddings.shape[1])

    correct = 0
    total = 0
    all_results = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            feats = vision_model.extract_features(images)
            if feats.dim() == 4:
                feats = feats.mean(dim=(2, 3))

            text_feats = class_text_embeddings[labels]
            outputs = fusion_model(feats, text_feats)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Save per-sample results for analysis
            for i in range(labels.size(0)):
                all_results.append({
                    "true_label": int(labels[i].item()),
                    "pred_label": int(preds[i].item()),
                })

    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}% ({correct}/{total})")

    os.makedirs("results", exist_ok=True)
    out_path = "results/fusion_test_results.json"
    with open(out_path, "w") as f:
        json.dump({"accuracy": accuracy, "results": all_results}, f, indent=2)

    print(f"Saved evaluation results to {out_path}")


if __name__ == "__main__":
    evaluate()
