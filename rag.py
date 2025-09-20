import os
import hashlib
import pickle
import chardet
import PyPDF2
import docx
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class RAGEngine:
    def __init__(self):
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            st.sidebar.error(f"Could not load embedding model: {e}")
            self.embedding_model = None
        self.documents = []
        self.embeddings = None

    def _chunk_text(self, text, max_length=500):
        words = text.split()
        for i in range(0, len(words), max_length):
            yield " ".join(words[i : i + max_length])

    def _hash_file(self, file_bytes):
        return hashlib.md5(file_bytes).hexdigest()

    def process_files(self, uploaded_files):
        self.documents = []
        all_embeddings = []

        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            file_hash = self._hash_file(file_bytes)
            cache_path = os.path.join(CACHE_DIR, f"{file_hash}.pkl")

            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    chunks, embeddings = pickle.load(f)
                self.documents.extend(chunks)
                all_embeddings.extend(embeddings)
                continue

            # --- Extract text ---
            file_contents = ""
            try:
                if uploaded_file.name.endswith(".txt"):
                    encoding = chardet.detect(file_bytes)["encoding"] or "utf-8"
                    file_contents = file_bytes.decode(encoding, errors="ignore")

                elif uploaded_file.name.endswith(".pdf"):
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        file_contents += page.extract_text() or ""

                elif uploaded_file.name.endswith(".docx"):
                    doc = docx.Document(uploaded_file)
                    file_contents = "\n".join([p.text for p in doc.paragraphs])

                elif uploaded_file.name.endswith(".md"):
                    file_contents = file_bytes.decode("utf-8", errors="ignore")

                # --- Chunk & embed ---
                chunks = []
                if file_contents.strip():
                    for chunk in self._chunk_text(file_contents):
                        chunks.append({"filename": uploaded_file.name, "content": chunk})

                    embeddings = self.embedding_model.encode([c["content"] for c in chunks])

                    with open(cache_path, "wb") as f:
                        pickle.dump((chunks, embeddings), f)

                    self.documents.extend(chunks)
                    all_embeddings.extend(embeddings)

            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")

        if all_embeddings:
            self.embeddings = np.array(all_embeddings)

    def retrieve(self, query, top_k=3):
        if not self.embeddings.any() or not self.embedding_model:
            return []

        query_emb = self.embedding_model.encode([query])
        sims = cosine_similarity(query_emb, self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if sims[idx] > 0.3:
                results.append(
                    {
                        "filename": self.documents[idx]["filename"],
                        "snippet": self.documents[idx]["content"][:200] + "...",
                        "similarity": float(sims[idx]),
                    }
                )
        return results
