import faiss

print("FAISS version:", faiss.__version__)

# Optional: simple test to create an index
dim = 768  # embedding dimension (same as InLegalBERT)
index = faiss.IndexFlatL2(dim)
print("Index created. Is trained?", index.is_trained)