import torch
import numpy as np

# Load the embeddings
embeddings = torch.load('data/processed/text_embeddings.pt')
print('=== TEXT EMBEDDINGS ANALYSIS ===')
print(f'Shape: {embeddings.shape}')
print(f'Device: {embeddings.device}')
print(f'Data type: {embeddings.dtype}')
print(f'Memory usage: {embeddings.element_size() * embeddings.numel()} bytes')
print()

# Check basic statistics
embeddings_np = embeddings.cpu().numpy()
print('=== STATISTICS ===')
print(f'Mean: {embeddings_np.mean():.4f}')
print(f'Std: {embeddings_np.std():.4f}')
print(f'Min: {embeddings_np.min():.4f}')
print(f'Max: {embeddings_np.max():.4f}')
print()

# Check if normalized (should be close to 1)
norms = np.linalg.norm(embeddings_np, axis=1)
print('=== NORMALIZATION CHECK ===')
print(f'Norms (should be ~1.0): {norms}')
print(f'All norms close to 1: {np.allclose(norms, 1.0, atol=1e-6)}')
print()

# Show sample embeddings
print('=== SAMPLE EMBEDDINGS ===')
sample_texts = ['River flowing through forest', 'Highway crossing urban area', 'Dense forest with mountains']
for i, text in enumerate(sample_texts):
    print(f'{i+1}. \"{text}\"')
    print(f'   First 5 values: {embeddings_np[i][:5]}')
    print(f'   Norm: {norms[i]:.6f}')
    print()