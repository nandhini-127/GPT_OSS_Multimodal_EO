import argparse
import os
from typing import List

import torch
from PIL import Image
from torchvision import transforms

from config import DEVICE
from vision.model import build_model
from fusion.fusion_model import build_fusion_model


def load_vision_model(checkpoint_path="checkpoints/vision_best.pth"):
    model = build_model().to(DEVICE)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Vision checkpoint not found: {checkpoint_path}")
    model.eval()
    return model


def load_fusion_model(vision_dim: int, text_dim: int, checkpoint_path="checkpoints/fusion_best.pth"):
    model = build_fusion_model(vision_dim=vision_dim, text_dim=text_dim, fusion_dim=512).to(DEVICE)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"Fusion checkpoint not found: {checkpoint_path}")
    model.eval()
    return model


def load_class_embeddings(path: str = "data/processed/class_text_embeddings.pt") -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Class text embeddings not found: {path}")
    return torch.load(path, map_location=DEVICE)


def get_val_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict(image_paths: List[str], class_names: List[str]):
    vision_model = load_vision_model()
    class_embeddings = load_class_embeddings()

    # Determine vision feature dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        feats = vision_model.extract_features(dummy)
        if feats.dim() == 4:
            feats = feats.mean(dim=(2, 3))
        vision_dim = feats.shape[-1]

    fusion_model = load_fusion_model(vision_dim=vision_dim, text_dim=class_embeddings.shape[1])

    transform = get_val_transform()

    predictions = []

    with torch.no_grad():
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(DEVICE)

            vision_feats = vision_model.extract_features(tensor)
            if vision_feats.dim() == 4:
                vision_feats = vision_feats.mean(dim=(2, 3))

            # Score all classes and pick best
            scores = fusion_model(vision_feats.repeat(len(class_embeddings), 1), class_embeddings)
            # because we repeated vision_feats, reshape
            scores = scores.view(-1, len(class_embeddings))

            probs = torch.softmax(scores, dim=1)
            best_idx = probs.argmax(dim=1).item()
            confidence = float(probs.max().item())

            predictions.append({
                "image": img_path,
                "predicted_class": class_names[best_idx],
                "predicted_index": int(best_idx),
                "confidence": confidence,
            })

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fusion inference on image(s)")
    parser.add_argument("--images", nargs="+", required=True, help="Path(s) to input image(s)")
    parser.add_argument("--classes", default=None, help="Optional path to class names text file")

    args = parser.parse_args()

    if args.classes is not None:
        with open(args.classes, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        # Default EuroSAT class ordering
        class_names = [
            "AnnualCrop",
            "Forest",
            "HerbaceousVegetation",
            "Highway",
            "Industrial",
            "Pasture",
            "PermanentCrop",
            "Residential",
            "River",
            "SeaLake",
        ]

    preds = predict(args.images, class_names)
    for p in preds:
        print(f"{p['image']} -> {p['predicted_class']} (idx={p['predicted_index']})")
