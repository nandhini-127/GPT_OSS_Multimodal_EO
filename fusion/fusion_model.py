"""Fusion model components.

This module provides a FusionModel wrapper that combines visual and textual
features via a simple feed-forward fusion network.
"""

from multimodal.multimodal_model import MultimodalModel


def build_fusion_model(vision_dim: int, text_dim: int, fusion_dim: int = 512):
    """Build a fusion model that combines vision + text embeddings."""
    return MultimodalModel(vision_dim=vision_dim, text_dim=text_dim, fusion_dim=fusion_dim)
