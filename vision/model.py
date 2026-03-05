import timm
import torch.nn as nn
from config import MODEL_NAME, NUM_CLASSES


class VisionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=NUM_CLASSES
        )

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        features = self.model.forward_features(x)

        # If transformer → shape (B, tokens, dim)
        if len(features.shape) == 3:
            features = features.mean(dim=1)

        return features


def build_model():
    return VisionModel()