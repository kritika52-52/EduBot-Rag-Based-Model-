"""
Section 6: Comparative Analysis.

Runs your evaluation questions (evaluation/eval_dataset.json) against every
vector store you've built (each = one embedding model + chunking strategy
combination) and produces a results table like the one in your project doc:

    Embedding Model | Chunking Type | Precision@3 | Faithfulness | Latency (sec)

Metric notes (simple, explainable versions suitable for a final-year report):
  - Precision@3: fraction of the top-3 retrieved chunks that contain at least
    one of the question's "source_keywords". This is a lightweight stand-in
    for human relevance judgement — mention this choice in your report.
  - Faithfulness: fraction of the generated answer's content words that also
    appear somewhere in the retrieved chunks (a simple overlap-based proxy).
    For a stronger version, ask a second LLM call "does this answer follow
    only from this context? yes/no" — mention both options in your report.
  - Latency: wall-clock seconds for retrieval + generation, averaged per
    question.

Before running, build one vector store per combination you want to test, e.g.:
    python embeddings/build_vector_store.py --chunks data/chunks/fixed_size.json --model minilm    --out vector_store/minilm_fixed
    python embeddings/build_vector_store.py --chunks data/chunks/semantic.json  --model minilm    --out vector_store/minilm_semantic
    python embeddings/build_vector_store.py --chunks data/chunks/fixed_size.json --model bge-small --out vector_store/bge_fixed
    python embeddings/build_vector_store.py --chunks data/chunks/semantic.json  --model bge-small --out vector_store/bge_semantic

Then list those store paths in STORES_TO_COMPARE below and run:
    python evaluation/evaluate.py
"""

import json
import time
from pathlib import Path

from retrieval.retriever import Retriever
from generation.llm import generate_answer

# Add every vector store folder you built, one entry per combination:
STORES_TO_COMPARE = [
    {"path": "vector_store/minilm_fixed", "embedding_model": "MiniLM", "chunking_type": "Fixed-size"},
    {"path": "vector_store/minilm_semantic", "embedding_model": "MiniLM", "chunking_type": "Semantic"},
    {"path": "vector_store/bge_fixed", "embedding_model": "BGE-small", "chunking_type": "Fixed-size"},
    {"path": "vector_store/bge_semantic", "embedding_model": "BGE-small", "chunking_type": "Semantic"},
]

EVAL_DATASET_PATH = Path("evaluation/eval_dataset.json")
TOP_K = 3


def precision_at_k(retrieved_chunks, source_keywords):
    if not source_keywords:
        return None
    hits = 0
    for chunk in retrieved_chunks:
        text_lower = chunk["text"].lower()
        if any(kw.lower() in text_lower for kw in source_keywords):
            hits += 1
    return hits / len(retrieved_chunks) if retrieved_chunks else 0.0


def faithfulness_score(answer: str, retrieved_chunks):
    context_words = set()
    for chunk in retrieved_chunks:
        context_words.update(w.lower().strip(".,!?") for w in chunk["text"].split())

    answer_words = [w.lower().strip(".,!?") for w in answer.split() if len(w) > 3]
    if not answer_words:
        return 0.0
    matched = sum(1 for w in answer_words if w in context_words)
    return matched / len(answer_words)


def evaluate_store(store_info: dict, eval_items: list):
    retriever = Retriever(store_info["path"])

    precisions, faithfulness_scores, latencies = [], [], []

    for item in eval_items:
        start = time.time()
        hits = retriever.retrieve(item["question"], top_k=TOP_K)
        answer = generate_answer(item["question"], hits)
        elapsed = time.time() - start

        p = precision_at_k(hits, item.get("source_keywords", []))
        f = faithfulness_score(answer, hits)

        if p is not None:
            precisions.append(p)
        faithfulness_scores.append(f)
        latencies.append(elapsed)

    return {
        "embedding_model": store_info["embedding_model"],
        "chunking_type": store_info["chunking_type"],
        "precision_at_3": round(sum(precisions) / len(precisions), 2) if precisions else None,
        "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 2),
        "latency_sec": round(sum(latencies) / len(latencies), 2),
    }


def main():
    if not EVAL_DATASET_PATH.exists():
        raise SystemExit(f"Create {EVAL_DATASET_PATH} first (see the template already provided).")

    eval_items = json.loads(EVAL_DATASET_PATH.read_text())

    results = []
    for store_info in STORES_TO_COMPARE:
        if not Path(store_info["path"]).exists():
            print(f"Skipping {store_info['path']} (not built yet)")
            continue
        print(f"Evaluating {store_info['embedding_model']} + {store_info['chunking_type']} ...")
        results.append(evaluate_store(store_info, eval_items))

    print("\n| Embedding Model | Chunking Type | Precision@3 | Faithfulness | Latency (sec) |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['embedding_model']} | {r['chunking_type']} | {r['precision_at_3']} "
              f"| {r['faithfulness']} | {r['latency_sec']} |")

    out_path = Path("evaluation/results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results also saved to {out_path}")


if __name__ == "__main__":
    main()
