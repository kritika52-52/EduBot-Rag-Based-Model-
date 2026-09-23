"""
Step 5: Retrieval.

Given a student's question, this loads a pre-built vector store (see
embeddings/build_vector_store.py) and returns the top-k most relevant chunks.

Used by both app.py (the live chatbot) and evaluation/evaluate.py (the
comparison experiments).
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, store_dir: str):
        store_dir = Path(store_dir)
        meta = json.loads((store_dir / "meta.json").read_text())

        self.model = SentenceTransformer(meta["model_name"])
        self.index = faiss.read_index(str(store_dir / "index.faiss"))
        with open(store_dir / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, question: str, top_k: int = 4):
        query_vector = self.model.encode([question], normalize_embeddings=True)
        query_vector = np.array(query_vector, dtype="float32")

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "similarity": float(score),
            })
        return results


if __name__ == "__main__":
    # Quick manual test:
    #   python -m retrieval.retriever
    r = Retriever("vector_store/minilm_fixed")
    for hit in r.retrieve("What is the minimum attendance required?"):
        print(f"[{hit['similarity']:.3f}] {hit['source']} -> {hit['text'][:120]}...")
