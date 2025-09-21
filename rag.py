# rag.py
import os
import hashlib
import pickle
from typing import List, Dict
import chardet
import PyPDF2
import docx
import streamlit as st

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_EMBED = True
except Exception:
    HAS_EMBED = False

CACHE_DIR = "cache_embeddings"
os.makedirs(CACHE_DIR, exist_ok=True)


class RAGEngine:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 300):
        self.chunk_size = chunk_size
        self.documents: List[Dict] = []
        self.embeddings = None
        self.embedding_model_name = embedding_model_name

        if HAS_EMBED:
            try:
                self.model = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                st.sidebar.error(f"Could not load embedding model {embedding_model_name}: {e}")
                self.model = None
        else:
            self.model = None
            st.sidebar.warning("Embedding dependencies not installed; RAG disabled.")

    def _hash_bytes(self, b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def _chunk_text(self, text: str):
        words = text.split()
        for i in range(0, len(words), self.chunk_size):
            yield " ".join(words[i:i + self.chunk_size])

    def process_files(self, uploaded_files) -> int:
        if not uploaded_files:
            return 0
        for f in uploaded_files:
            try:
                raw = f.getvalue()
                file_hash = self._hash_bytes(raw)
                cache_file = os.path.join(CACHE_DIR, f"{file_hash}.pkl")
                if os.path.exists(cache_file):
                    with open(cache_file, "rb") as fh:
                        chunks = pickle.load(fh)
                    self.documents.extend(chunks)
                    continue

                text = ""
                name = f.name.lower()
                if name.endswith(".txt"):
                    encoding = chardet.detect(raw).get("encoding") or "utf-8"
                    text = raw.decode(encoding, errors="ignore")
                elif name.endswith(".pdf"):
                    reader = PyPDF2.PdfReader(f)
                    for p in reader.pages:
                        text += (p.extract_text() or "") + "\n"
                elif name.endswith(".docx"):
                    doc = docx.Document(f)
                    text = "\n".join([p.text for p in doc.paragraphs])
                elif name.endswith(".md"):
                    text = raw.decode("utf-8", errors="ignore")
                else:
                    text = raw.decode("utf-8", errors="ignore")

                chunks = []
                for chunk in self._chunk_text(text):
                    chunks.append({"filename": f.name, "content": chunk})

                with open(cache_file, "wb") as fh:
                    pickle.dump(chunks, fh)

                self.documents.extend(chunks)
            except Exception as e:
                st.sidebar.error(f"Error processing {f.name}: {e}")

        if self.model and self.documents:
            texts = [d["content"] for d in self.documents]
            try:
                self.embeddings = self.model.encode(texts, show_progress_bar=False)
            except Exception as e:
                st.sidebar.error(f"Embedding error: {e}")
                self.embeddings = None

        return len(self.documents)

    def retrieve(self, query: str, top_k: int = 3):
        """Return top_k similar chunks as list of dicts {filename, snippet, similarity}"""
        if not self.model or self.embeddings is None or len(self.documents) == 0:
            return []
        q_emb = self.model.encode([query])
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        idxs = sims.argsort()[::-1][:top_k]

        results = []
        for i in idxs:
            results.append({
                "filename": self.documents[i]["filename"],
                "snippet": (self.documents[i]["content"][:300] + "...") if len(self.documents[i]["content"]) > 300 else self.documents[i]["content"],
                "similarity": float(sims[i])
            })
        return results
