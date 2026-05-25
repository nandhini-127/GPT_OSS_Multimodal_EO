import timm
import torch.nn as nn
from config import MODEL_NAME, NUM_CLASSES


class VisionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 🔥 Backbone (no classifier)
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        # 🔥 Your classifier head
        self.classifier = nn.Linear(self.backbone.num_features, NUM_CLASSES)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def extract_features(self, x):
        return self.backbone(x)


def build_model():
    return VisionModel()