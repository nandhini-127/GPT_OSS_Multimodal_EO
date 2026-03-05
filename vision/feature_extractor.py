import torch
import numpy as np
from torch.utils.data import DataLoader
from vision.model import build_model
from vision.dataset import get_dataloaders
from config import DEVICE

def extract_embeddings():

    _, _, test_loader = get_dataloaders()

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(
        "checkpoints/vision/best_model.pth",
        map_location=DEVICE
    ))

    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)

            features = model.extract_features(images)
            all_embeddings.append(features.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    np.save("data/processed/eo_embeddings.npy", embeddings)

    print("✅ EO embeddings saved successfully!")


if __name__ == "__main__":
    extract_embeddings()