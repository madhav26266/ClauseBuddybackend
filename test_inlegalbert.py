from transformers import AutoTokenizer, AutoModel
import torch

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModel.from_pretrained("law-ai/InLegalBERT")

def get_embeddings(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    pooled = embeddings.mean(dim=1)  # single vector
    return pooled

# Example usage
text = "This is a legal document describing a court ruling."
emb = get_embeddings(text)
print("Embedding shape:", emb.shape)  # Expected: torch.Size([1, 768])
print("First 5 values of embedding:", emb[0][:5])