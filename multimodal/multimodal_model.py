import torch
import torch.nn as nn

class MultimodalModel(nn.Module):
    def __init__(self, vision_dim=1024, text_dim=768, fusion_dim=512):
        super().__init__()

        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        self.fusion_layer = nn.Linear(fusion_dim * 2, fusion_dim)
        self.classifier = nn.Linear(fusion_dim, 10)

    def forward(self, vision_feat, text_feat):
        v = self.vision_proj(vision_feat)
        t = self.text_proj(text_feat)

        combined = torch.cat([v, t], dim=1)
        fused = torch.relu(self.fusion_layer(combined))

        output = self.classifier(fused)
        return output