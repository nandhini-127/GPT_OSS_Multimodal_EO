from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

def load_llm(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load pre-trained LLM and tokenizer.

    Returns:
        tokenizer: Transformer tokenizer
        model: Sentence transformer model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SentenceTransformer(model_name)
    return tokenizer, model