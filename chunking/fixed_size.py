"""
Step 2a: Fixed-size chunking.

Reads every PDF/DOCX/TXT in data/raw/, extracts the text, and splits it into
fixed-size chunks (default ~400 words) with a small overlap so we don't cut
sentences in half at chunk boundaries.

Output: data/chunks/fixed_size.json
Each chunk looks like:
{
  "chunk_id": "attendance_policy.pdf_0",
  "source": "attendance_policy.pdf",
  "text": "..."
}

Run:
    python chunking/fixed_size.py
"""

import os
import json
from pathlib import Path
from pypdf import PdfReader
import docx

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/chunks/fixed_size.json")
CHUNK_SIZE_WORDS = 400
OVERLAP_WORDS = 50


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx(path: Path) -> str:
    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".txt":
        return read_txt(path)
    return ""


def chunk_text(text: str, size: int, overlap: int):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        start = end - overlap  # step forward, but re-include the overlap
        if end >= len(words):
            break
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
            print(f"  WARNING: no extractable text in {file_path.name} (might be a scanned PDF).")
            continue
        pieces = chunk_text(text, CHUNK_SIZE_WORDS, OVERLAP_WORDS)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{file_path.name}_{i}",
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
