from llm.llm_loader import load_llm

class TextTokenizer:
    """
    Optional wrapper if you want separate tokenization utilities.
    """
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        self.tokenizer, _ = load_llm(model_name)

    def tokenize(self, text_list, max_length=512):
        return self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )