# Student Handbook RAG Assistant

## 1. Project Overview

An AI-powered assistant that answers student questions about the Zaio
coding bootcamp handbooks. Students ask questions in plain English; the
system searches the handbook PDFs, retrieves the most relevant passages,
and generates an answer grounded **only** in the handbook content. If the
handbook does not contain the answer, the assistant says so instead of
guessing.

## 2. Problem Statement

Bootcamp handbooks are long PDFs (fees, schedules, policies, grading).
Students repeatedly ask the same questions in chat. This assistant gives
instant, accurate, page-cited answers — and refuses to invent information
that is not in the handbook.

## 3. Technologies Used

| Technology | Role |
|---|---|
| Python 3.12 | Main language |
| FastAPI | REST API (`POST /ask`, `GET /health`) |
| PyMuPDF (`fitz`) | PDF text extraction, page-by-page |
| LangChain (`langchain-text-splitters`) | Recursive character text splitting |
| Sentence Transformers (`all-MiniLM-L6-v2`) | Text embeddings (384-dim) |
| ChromaDB (persistent) | Vector database for chunk embeddings |
| Pydantic / pydantic-settings | Request validation + env config |
| Groq API (`openai/gpt-oss-120b`) | LLM answer generation (OpenAI-compatible) |
| Pytest + FastAPI TestClient | Automated tests (LLM calls mocked) |

## 4. Project Structure

```text
student-handbook-rag/
├── app/
│   ├── __init__.py            # truststore SSL injection (corporate proxies)
│   ├── main.py                # FastAPI app: GET /health, POST /ask
│   ├── config.py              # Settings from env vars / .env
│   ├── ingestion/
│   │   ├── pdf_loader.py      # PyMuPDF extraction, page numbers, multi-PDF
│   │   ├── text_splitter.py   # LangChain chunking + metadata (Chunk)
│   │   └── embedder.py        # SentenceTransformer wrapper (lazy load)
│   ├── retrieval/
│   │   ├── vector_store.py    # Persistent ChromaDB + deterministic IDs
│   │   └── retriever.py       # Question embedding + threshold filter
│   ├── generation/
│   │   └── answer_generator.py# Groq call, grounded prompt, fallback
│   └── schemas/
│       └── request_models.py  # AskRequest / AskResponse (Pydantic)
├── data/
│   ├── fullstack-web-dev-handbook.pdf      # Handbook #1
│   └── fullstack-ai-engineer-handbook.pdf  # Handbook #2
├── chroma_db/                 # Persistent vector DB (generated, gitignored)
├── tests/                     # 31 automated tests
├── scripts/
│   └── ingest_handbook.py     # Indexing CLI
├── .env                       # Your secrets (gitignored — never commit)
├── .env.example               # Template for .env
├── .gitignore
├── requirements.txt
└── README.md
```

## 5. Installation

Requires **Python 3.12** (3.14 lacks wheels for several pinned deps).

```powershell
cd student-handbook-rag
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

> **Note:** several packages are pinned to older versions
> (`regex`, `numpy`, `scipy`, `scikit-learn`, `grpcio`, `xxhash`) because
> this machine's Application Control policy blocks the newest releases'
> native DLLs. Keep the pins.

## 6. Environment Configuration

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set your LLM key. Supported providers: `groq` (used here),
`openai`, `anthropic`, `google`, `ollama`. Groq example:

```env
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=gsk_...
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

`.env` is gitignored — **never commit real keys**.

## 7. Adding the Handbook PDF(s)

Place PDF(s) in `data/`. The loader picks up **every** `*.pdf` in that
folder, so multiple handbooks can be indexed together — each chunk stores
its `document_name` so answers cite the correct book. Page numbers are the
original PDF page numbers.

## 8. Running Ingestion (indexing)

```powershell
python scripts/ingest_handbook.py
```

Output (actual run):

```text
Loading PDFs from: ...\student-handbook-rag\data
  -> 48 pages with text
  -> 73 chunks
  -> embedding model: all-MiniLM-L6-v2 (dim 384)
  -> 73 embeddings generated
  -> 73 records upserted into ChromaDB
  -> collection 'student_handbook' now holds 73 chunks
```

Re-running is safe: chunk IDs are deterministic (SHA-1 of
doc+page+text), so records are **upserted, not duplicated**. Note:
changing the chunking logic requires wiping `chroma_db/` once so stale
chunks are removed. The API never re-indexes at startup — it reads the
persistent `chroma_db/`.

## 9. Starting the API

```powershell
uvicorn app.main:app --reload --port 8000
```

- `GET  http://localhost:8000/health` → `{"status": "ok"}`
- `POST http://localhost:8000/ask`

## 10. Testing the API

```powershell
curl -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"question": "When are the live classes held?"}'
```

Example response (actual):

```json
{
  "answer": "Live classes are scheduled for Wednesdays from 6 pm to 8 pm and Mondays from 9 am to 11 am during the first 12 weeks of the bootcamp. After those 12 weeks, the schedule reduces to a single live class each Monday from 9 am to 11 am. All sessions are recorded.",
  "source": "fullstack-web-dev-handbook - Page 11"
}
```

Fallback response (actual, for unanswerable questions):

```json
{
  "answer": "I could not find that information in the available knowledge base.",
  "source": null
}
```

## 11. How the RAG Process Works

1. **Ingest** (`scripts/ingest_handbook.py`): PyMuPDF extracts text per
   page → LangChain splits each page into ~800-char overlapping chunks →
   Sentence Transformer embeds each chunk (384-dim, normalized) → chunks +
   embeddings + metadata upserted into persistent ChromaDB.
2. **Retrieve** (`Retriever`): the question is embedded with the *same*
   model → ChromaDB cosine search returns top-20 chunks → chunks below the
   0.3 similarity threshold are discarded, so unrelated questions get no
   context.
3. **Generate** (`AnswerGenerator`): retrieved excerpts are placed in a
   prompt that forbids outside knowledge; Groq answers concisely. If no
   context survives the threshold, or the model reports the answer missing,
   the exact fallback message is returned with `source: null`.

## 12. Running Automated Tests

```powershell
pytest tests/ -v
```

31 tests covering: PDF loading/extraction/page numbers/empty pages,
chunking metadata, embedding dimensions, ChromaDB storage + dedup,
retrieval accuracy + threshold, API success/validation/error paths, and
all fallback behaviours. LLM calls are stubbed — tests never use the paid
API; embeddings run locally (free).

## 13. Manual Test Results (actual runs)

| # | Question | Source | Answer (abridged) |
|---|----------|--------|-------------------|
| 1 | When does the bootcamp start and how long does it run? | ai-engineer p7 | Begins 30 March 2026; 7 months (AI) / 6 months (web-dev) |
| 2 | What LMS does the bootcamp use? | web-dev p13 | Google Classroom for lectures and assignments |
| 3 | How are final grades determined? | ai-engineer p14 | Final Project 35%, Assignments 25%, Coding Challenges 25% |
| 4 | How do I access my calendar? | web-dev p10 | Create zaio.io account, add email to Google sheet by 31 Mar |
| 5 | What are the live class times? | web-dev p11 | Mon 9–11am & Wed 6–8pm first 12 weeks, then Mon only |
| 6 | Who is the lead instructor? | web-dev p15 | Kago Kola |
| 7 | What happens if I miss a live class? | web-dev p11 | Watch recordings; tutors on standby for doubts |
| 8 | What is the attendance requirement? | — | Fallback (no formal policy in handbook — correct) |
| 9 | What is the refund policy? | — | Fallback (not in handbook — correct) |
| 10 | Do students get a laptop provided? | — | Fallback (not in handbook — correct) |

## 14. Limitations and Future Improvements

- **Embedding model**: `all-MiniLM-L6-v2` is lightweight; a larger model
  (e.g. `bge-large`) could improve retrieval precision at higher cost.
- **Retrieval**: dense-only search; hybrid keyword+dense (BM25) would help
  exact-term queries like "LMS".
- **Chunking**: fixed 800-char chunks; semantic/section-aware splitting
  could preserve policy boundaries better.
- **LLM**: single provider (Groq); the generator is structured to add
  OpenAI/Anthropic/Google/Ollama providers.
- **Multi-handbook**: both handbooks share one collection; per-document
  filtering or separate collections could be added if needed.
- **No frontend**: API-only by design (n8n-ready JSON in/out).
