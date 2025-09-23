# microservice.py
import os
import io
from typing import List

import torch
import numpy as np
import faiss
import PyPDF2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from gemini_client import query_gemini
from dotenv import load_dotenv
load_dotenv()



class GeminiQuery(BaseModel):
    query: str
    top_k_files: int = 3

index = faiss.IndexFlatL2(768)

# Convert text to vector for FAISS
def embed_text(text: str) -> np.ndarray:
    """Return NumPy vector embedding for FAISS"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # shape: (1, 768)
    return embeddings.cpu().numpy().astype("float32")[0]  # returns 1D vector



# --------------------------
# CPU-safe setup
# --------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Ignore OpenMP errors
torch.set_num_threads(1)                      # PyTorch single-thread
faiss.omp_set_num_threads(1)                  # FAISS single-thread

async def upload_pdfs(files: List[UploadFile] = File(...)):
    global current_index, text_mapping, file_mapping
    results = []
# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(title="InLegalBERT + FAISS Microservice")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:5000"] for Node backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Load InLegalBERT
# --------------------------
MODEL_NAME = "law-ai/InLegalBERT"
API_KEY = os.getenv("HUGGINGFACE_API_KEY", "YOUR_HUGGINGFACE_API_KEY_HERE")

print("Loading InLegalBERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
print("✅ InLegalBERT model loaded successfully!")

def get_embedding(text: str) -> np.ndarray:
    """Convert text to embedding using InLegalBERT"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # (1, hidden_size)
    return embeddings.numpy().astype("float32")  # FAISS requires float32

# --------------------------
# FAISS setup
# --------------------------
embedding_dim = 768
faiss_index = faiss.IndexFlatL2(embedding_dim)
file_mapping = {}  # maps index position → filename
current_index = 0

# --------------------------
# Legal summary
# --------------------------
def create_legal_summary(text: str):
    key_terms = ["agreement", "contract", "employee", "company", "confidentiality",
                 "terms", "conditions", "employment", "salary", "benefits", "termination"]
    found_terms = [term for term in key_terms if term.lower() in text.lower()]
    doc_type = "employment agreement" if "employee" in found_terms else "legal contract"

    summary = (
        f"Legal document analysis: {len(text)} chars, {len(text.split())} words. "
        f"Document type: {doc_type}. "
        f"Key legal terms: {', '.join(found_terms[:5])}."
    )
    return summary

# --------------------------
# Request models
# --------------------------
class TextRequest(BaseModel):
    text: str
    filename: str

class QueryRequest(BaseModel):
    query: str
    k: int = 3

# --------------------------
# Endpoints
# --------------------------

@app.get("/")
async def root():
    return {"message": "Hello, Backend is running 🚀"}

@app.post("/process-text")
async def process_text(req: TextRequest):
    global current_index
    
    # Create summary
    summary = create_legal_summary(req.text)
    
    # Add to FAISS
    embedding = get_embedding(req.text)
    faiss_index.add(embedding)
    file_mapping[current_index] = req.filename
    current_index += 1
    
    return {"summary": summary, "indexed": True}

# Globals
file_mapping = {}
text_mapping = {}
current_index = 0

@app.post("/api/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global current_index
    results = []

    for file in files:
        try:
            contents = await file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

            # --- Split into chunks for FAISS ---
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]

            for chunk in chunks:
                # Create embedding for each chunk
                embedding = get_embedding(chunk).astype("float32").reshape(1, -1)

                # Add to FAISS
                faiss_index.add(embedding)
                file_mapping[current_index] = file.filename
                text_mapping[current_index] = chunk   # store chunk text
                current_index += 1

            results.append({
                "filename": file.filename,
                "chunks_indexed": len(chunks)
            })

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"summaries": results}

@app.post("/api/query")
async def query_embeddings(req: QueryRequest):
    query = req.query
    k = req.k

    if faiss_index.ntotal == 0:
        return {"error": "No documents indexed yet."}

    embedding = get_embedding(query).astype("float32").reshape(1, -1)
    D, I = faiss_index.search(embedding, k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append({
            "filename": file_mapping[idx],
            "distance": float(dist),
            "text": text_mapping[idx]   # ✅ now works
        })

    return {"results": results}


@app.post("/api/gemini-answer")
async def gemini_answer(req: GeminiQuery):
    try:
        if faiss_index.ntotal == 0:
            return {"error": "No documents indexed yet."}

        # 1️⃣ Convert query to embedding (same function as indexing)
        query_vector = get_embedding(req.query).astype("float32").reshape(1, -1)

        # 2️⃣ Search FAISS index
        distances, indices = faiss_index.search(query_vector, req.top_k_files)

        # 3️⃣ Filter valid indices
        valid_indices = [i for i in indices[0] if i != -1 and i in text_mapping]

        if not valid_indices:
            return {"error": "No relevant chunks found in indexed documents."}

        # 4️⃣ Retrieve corresponding chunks
        retrieved_chunks = [text_mapping[i] for i in valid_indices]

        print(f"Retrieved {len(retrieved_chunks)} chunks for Gemini.")
        for idx, chunk in zip(valid_indices, retrieved_chunks):
            print(f"Index {idx}: {chunk[:100]}...")  # preview first 100 chars

        # 5️⃣ Build context
        context = "\n\n".join(retrieved_chunks)

        # 6️⃣ Construct prompt for Gemini
        prompt = f"""
You are a legal assistant. Use the following documents to answer the user query in a very simple, easy to understand, layman language.

Documents:
{context}

User Question:
{req.query}

Provide a clear, concise answer and summarize key clauses if relevant.
"""

        # 7️⃣ Call Gemini
        answer = query_gemini(prompt, max_tokens=300)

        return {"answer": answer, "retrieved_chunks": len(retrieved_chunks)}

    except Exception as e:
        return {"error": str(e)}
    


# --------------------------
# Run server
# --------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting InLegalBERT + FAISS Microservice on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
