import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from vision.model import build_model
from vision.dataset import get_dataloaders
from config import DEVICE


def get_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_vision_model(checkpoint_path: str = "checkpoints/vision_best.pth"):
    model = build_model().to(DEVICE)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Vision checkpoint not found: {checkpoint_path}")
    model.eval()
    return model


def extract_embeddings_from_image(image: Image.Image, model=None):
    """Return a single feature vector for a single image."""
    if model is None:
        model = load_vision_model()

    transform = get_transform()
    with torch.no_grad():
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        features = model.extract_features(tensor)
        if features.dim() == 4:
            features = features.mean(dim=(2, 3))
    return features.cpu().numpy()
def extract_single_embedding(image, model, device):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    x = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.extract_features(x)

        # ✅ FIX for transformer output
        if emb.dim() == 3:   # (B, tokens, dim)
            emb = emb.mean(dim=1)

        elif emb.dim() == 4: # (B, C, H, W)
            emb = emb.mean(dim=(2, 3))

    return emb.squeeze(0)



def extract_embeddings():

    _, _, test_loader = get_dataloaders()

    model = load_vision_model()

    all_embeddings = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)

            features = model.extract_features(images)
            all_embeddings.append(features.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/eo_embeddings.npy", embeddings)

    print("✅ EO embeddings saved successfully!")


if __name__ == "__main__":
    extract_embeddings()