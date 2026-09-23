"""
Step 3 & 4: Embedding + Vector Database.

Loads a chunk file produced by chunking/fixed_size.py or chunking/semantic.py,
converts each chunk to a vector using a Sentence-Transformers model, and stores
everything in a FAISS index on disk so it can be searched later.

Supports the 2-3 embedding models mentioned in your project doc so you can
build a separate vector store per (embedding_model, chunking_type) combination
for the comparison experiments in Section 6.

Run:
    python embeddings/build_vector_store.py --chunks data/chunks/fixed_size.json --model minilm --out vector_store/minilm_fixed
    python embeddings/build_vector_store.py --chunks data/chunks/semantic.json  --model bge-small --out vector_store/bge_semantic
"""

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Map short names -> actual HuggingFace model ids (all free to download)
MODEL_MAP = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


def load_chunks(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks, model_name: str):
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks ...")
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    vectors = np.array(vectors, dtype="float32")

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product = cosine sim (vectors are normalized)
    index.add(vectors)

    return index, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="Path to chunk JSON file")
    parser.add_argument("--model", required=True, choices=MODEL_MAP.keys())
    parser.add_argument("--out", required=True, help="Output folder for the vector store")
    args = parser.parse_args()

    chunks = load_chunks(Path(args.chunks))
    model_name = MODEL_MAP[args.model]
    index, model = build_index(chunks, model_name)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"model_key": args.model, "model_name": model_name, "num_chunks": len(chunks)}, f, indent=2)

    print(f"\nVector store saved to {out_dir}/  ({len(chunks)} chunks, model={model_name})")


if __name__ == "__main__":
    main()
