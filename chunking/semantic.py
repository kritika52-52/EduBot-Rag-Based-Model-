"""
Step 2b: Semantic (paragraph-based) chunking — the second strategy you compare
against fixed_size.py in your Section 6 experiments.

Instead of cutting every N words, this splits on paragraph breaks and then
merges small paragraphs together until each chunk is roughly TARGET_WORDS long,
so a chunk tends to hold one complete idea rather than a random slice of text.

Output: data/chunks/semantic.json (same format as fixed_size.json)

Run:
    python chunking/semantic.py
"""

import json
from pathlib import Path
from chunking.fixed_size import load_document_text, RAW_DIR  # reuse the readers

OUT_PATH = Path("data/chunks/semantic.json")
TARGET_WORDS = 400
MIN_WORDS = 80  # merge paragraphs smaller than this into the next one


def split_into_paragraphs(text: str):
    raw_paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return raw_paragraphs


def merge_into_semantic_chunks(paragraphs, target_words: int, min_words: int):
    chunks = []
    buffer = []
    buffer_word_count = 0

    for para in paragraphs:
        para_word_count = len(para.split())
        buffer.append(para)
        buffer_word_count += para_word_count

        if buffer_word_count >= target_words:
            chunks.append(" ".join(buffer))
            buffer = []
            buffer_word_count = 0

    if buffer and buffer_word_count >= min_words:
        chunks.append(" ".join(buffer))
    elif buffer and chunks:
        # too small on its own — merge tail into the last chunk
        chunks[-1] += " " + " ".join(buffer)
    elif buffer:
        chunks.append(" ".join(buffer))

    return chunks


def main():
    if not RAW_DIR.exists():
        raise SystemExit(f"Put your PDFs/DOCX/TXT files in {RAW_DIR}/ first.")

    all_chunks = []
    files = [f for f in RAW_DIR.iterdir() if f.suffix.lower() in (".pdf", ".docx", ".txt")]
    if not files:
        raise SystemExit(f"No documents found in {RAW_DIR}/. Add your college documents there.")

    for file_path in files:
        print(f"Reading {file_path.name} ...")
        text = load_document_text(file_path)
        if not text.strip():
            print(f"  WARNING: no extractable text in {file_path.name}.")
            continue
        paragraphs = split_into_paragraphs(text)
        pieces = merge_into_semantic_chunks(paragraphs, TARGET_WORDS, MIN_WORDS)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{file_path.name}_sem_{i}",
                "source": file_path.name,
                "text": piece,
            })
        print(f"  -> {len(pieces)} chunks")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_chunks)} total chunks written to {OUT_PATH}")


if __name__ == "__main__":
    main()
