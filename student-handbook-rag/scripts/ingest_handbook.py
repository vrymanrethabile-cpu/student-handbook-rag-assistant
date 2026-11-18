"""CLI script: index the Student Handbook PDF(s) into ChromaDB.

Usage (from the project root, with the venv activated):

    python scripts/ingest_handbook.py

Steps:
1. Load every PDF from the data/ directory (page-by-page).
2. Split each page into overlapping chunks.
3. Generate embeddings with the configured Sentence Transformer.
4. Upsert into the persistent ChromaDB collection.

Re-running is safe: deterministic chunk IDs mean records are updated
in place rather than duplicated, and unchanged chunks are not
re-embedded unnecessarily at the storage level.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/ingest_handbook.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.ingestion.embedder import Embedder  # noqa: E402
from app.ingestion.pdf_loader import load_all_handbooks  # noqa: E402
from app.ingestion.text_splitter import split_pages  # noqa: E402
from app.retrieval.vector_store import VectorStore  # noqa: E402


def main() -> None:
    data_dir = settings.project_root / "data"
    print(f"Loading PDFs from: {data_dir}")

    pages = load_all_handbooks(data_dir)
    print(f"  -> {len(pages)} pages with text")

    chunks = split_pages(pages)
    print(f"  -> {len(chunks)} chunks")

    embedder = Embedder()
    print(f"  -> embedding model: {embedder.model_name} (dim {embedder.dimension})")
    embeddings = embedder.embed_documents([c.text for c in chunks])
    print(f"  -> {len(embeddings)} embeddings generated")

    store = VectorStore()
    written = store.upsert_chunks(chunks, embeddings)
    print(f"  -> {written} records upserted into ChromaDB")
    print(f"  -> collection '{settings.chroma_collection_name}' now holds {store.count} chunks")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
