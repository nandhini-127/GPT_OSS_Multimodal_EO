import os
import torch
import numpy as np

class TextEmbeddingExtractor:
    """
    Extract embeddings from text using a simple approach
    """

    def __init__(self, model_name="mock-embeddings", device=None):
        # Auto-detect device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Loading model: {model_name} ...")
        # For now, use a simple mock implementation
        self.embedding_dim = 384  # Similar to MiniLM
        print("Model loaded successfully!")

        # EuroSAT class descriptions
        self.class_descriptions = {
            "AnnualCrop": "Agricultural field with annual crops, seasonal planting patterns",
            "Forest": "Dense forest area with tall trees and canopy cover",
            "HerbaceousVegetation": "Grassland and herbaceous vegetation, natural meadows",
            "Highway": "Major road infrastructure, transportation network",
            "Industrial": "Industrial buildings and manufacturing facilities",
            "Pasture": "Managed grassland for livestock grazing",
            "PermanentCrop": "Orchards and vineyards, perennial agricultural land",
            "Residential": "Urban residential areas with houses and buildings",
            "River": "Natural water bodies, rivers and streams",
            "SeaLake": "Large water bodies, seas and lakes"
        }

    def get_class_embeddings(self):
        """Get embeddings for all EuroSAT classes"""
        class_texts = list(self.class_descriptions.values())
        return self.get_embedding(class_texts)

    def get_embedding(self, text_list):
        """
        Convert a list of text strings into embeddings.
        For demonstration, generate random embeddings based on text length
        """
        embeddings = []
        for text in text_list:
            # Create a simple deterministic embedding based on text
            seed = hash(text) % 2**32
            np.random.seed(seed)
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return torch.tensor(np.array(embeddings), device=self.device)


def get_text_embedding(text):
    """Get embedding for a single text string."""
    extractor = TextEmbeddingExtractor()
    return extractor.get_embedding([text])[0]  # returns tensor


if __name__ == "__main__":
    # Generate class-level text embeddings for EuroSAT
    extractor = TextEmbeddingExtractor()

    print("Generating class-level text embeddings...")
    class_embeddings = extractor.get_class_embeddings()
    print("✅ Class embeddings generated!")
    print("Class embedding shape:", class_embeddings.shape)

    # Save class embeddings
    os.makedirs("data/processed", exist_ok=True)
    torch.save(class_embeddings, "data/processed/class_text_embeddings.pt")
    print("✅ Class text embeddings saved to data/processed/class_text_embeddings.pt")

    # For demonstration, also save sample embeddings
    sample_texts = [
        "River flowing through forest",
        "Highway crossing urban area",
        "Dense forest with mountains"
    ]

    print("\nGenerating sample embeddings...")
    sample_embeddings = extractor.get_embedding(sample_texts)
    torch.save(sample_embeddings, "data/processed/text_embeddings.pt")
    print("✅ Sample text embeddings saved to data/processed/text_embeddings.pt")

    # Example text descriptions
    sample_texts = [
        "River flowing through forest",
        "Highway crossing urban area",
        "Dense forest with mountains"
    ]

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = extractor.get_embedding(sample_texts)
    print("✅ Embeddings generated!")
    print("Embedding shape:", embeddings.shape)
    print("First embedding vector (truncated):", embeddings[0][:10])

    # Save embeddings
    os.makedirs("data/processed", exist_ok=True)
    torch.save(embeddings, "data/processed/text_embeddings.pt")
    print("✅ Text embeddings saved to data/processed/text_embeddings.pt")