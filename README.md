# EduBot — RAG-Based Academic Query Assistant

This is a working starter codebase for your final year project, matching the pipeline in your
project document: Chunking → Embedding → Vector Store → Retrieval → Generation → Streamlit UI,
plus an evaluation script for the comparative research section.

## 1. Folder structure

```
EduBot/
├── data/
│   ├── raw/              <- put your original PDFs/DOCX here
│   └── chunks/            <- auto-generated chunk JSON files land here
├── chunking/
│   ├── fixed_size.py
│   └── semantic.py
├── embeddings/
│   └── build_vector_store.py
├── retrieval/
│   └── retriever.py
├── generation/
│   └── llm.py
├── evaluation/
│   ├── eval_dataset.json     <- you fill this in (see Section 4 below)
│   └── evaluate.py
├── app.py                    <- Streamlit chat UI (final deliverable)
├── requirements.txt
└── README.md
```

## 2. Setup in VS Code

1. Open the `EduBot` folder in VS Code (`File > Open Folder`).
2. Create a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Get a **free Gemini API key**: https://aistudio.google.com/apikey
   Set it as an environment variable (don't hardcode it in code):
   ```bash
   # Windows (PowerShell)
   setx GEMINI_API_KEY "your_key_here"
   # Mac/Linux
   export GEMINI_API_KEY="your_key_here"
   ```
   (If you'd rather run a fully free/local model instead of an API, see "Using Ollama"
   at the bottom of `generation/llm.py`.)

## 3. What data / dataset do you need?

Your "dataset" is **not a CSV of numbers** — it's a **document knowledge base**. Collect real
documents from your own college (ask your HOD/office, or use what's on your college website):

| Document type | Example |
|---|---|
| Academic handbook / regulations | Attendance rules, grading policy, credit system |
| Syllabus PDFs | Subject-wise syllabus for each semester |
| Exam rules | Internal marks, re-exam rules, malpractice policy |
| Timetable | Class schedules, exam schedules |
| Fee / scholarship circulars | Fee deadlines, scholarship eligibility |
| FAQs (if any) | Previously answered student questions |

**Minimum for a good demo:** 5–10 PDFs (handbook + 2–3 syllabus files + exam rules) totaling
50–150 pages. That's enough to produce a few hundred chunks, which is plenty for a working demo
and for the comparison experiments in Section 6 of your report.

Put all of these inside `data/raw/`.

**Format tip:** if a document is a scanned/image PDF (no selectable text), you'll need OCR first
(e.g. `pytesseract`) — ask if you run into this, it's a common final-week surprise.

## 4. Data you need to *create yourself*: the evaluation set

For the comparative analysis (Section 6 of your project doc — Precision@3, Faithfulness,
Latency), you need a small **question–answer test set** written by your team, e.g. 20–30 pairs:

```json
[
  {
    "question": "What is the minimum attendance percentage required?",
    "expected_answer": "75%",
    "source_chunk_keywords": ["attendance", "75", "condonation"]
  }
]
```

This goes in `evaluation/eval_dataset.json`. Write these by hand by reading your own handbook —
that's normal and expected for a project like this; it becomes your "ground truth."

## 5. Run order

```bash
# 1. Chunk your documents (choose one strategy, or run both to compare)
python chunking/fixed_size.py
python chunking/semantic.py

# 2. Build the vector store (repeat once per embedding model you want to compare)
python embeddings/build_vector_store.py --chunks data/chunks/fixed_size.json --model bge-small --out vector_store/bge_fixed

# 3. Run the evaluation (compares combinations, produces the results table)
python evaluation/evaluate.py

# 4. Launch the chatbot UI
streamlit run app.py
```

Each script has a short docstring at the top explaining exactly what it does — read those first.
