# test_llm.py
from transformers import AutoTokenizer, AutoModel
import torch

print("Starting test...")

# Use a lightweight model to make sure it runs on CPU
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

texts = ["River flowing through forest", "Highway crossing urban area"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)

print("Embedding shape:", embeddings.shape)
print("First embedding vector (truncated):", embeddings[0][:10])