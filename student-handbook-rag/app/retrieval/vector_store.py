"""Persistent ChromaDB vector store.

Uses a PersistentClient so indexed chunks survive restarts — the API
never re-embeds the handbook at startup. Chunk IDs are deterministic
(SHA-1 of document+page+text) so re-running ingestion upserts in place
instead of creating duplicates.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import settings
from app.ingestion.text_splitter import Chunk


@dataclass
class SearchResult:
    """One retrieved chunk with its similarity score and metadata."""

    text: str
    score: float  # cosine similarity, higher = more relevant
    document_name: str
    page_number: int
    source: str


def _chunk_id(chunk: Chunk) -> str:
    """Deterministic ID so re-ingestion updates rather than duplicates."""
    key = f"{chunk.document_name}|{chunk.page_number}|{chunk.text}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]


class VectorStore:
    """Wrapper around a persistent ChromaDB collection."""

    def __init__(self, db_path: str | None = None, collection_name: str | None = None):
        self._client = chromadb.PersistentClient(
            path=str(db_path or settings.chroma_db_absolute)
        )
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name or settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine distance = 1 - similarity
        )

    @property
    def collection(self) -> Collection:
        return self._collection

    @property
    def count(self) -> int:
        return self._collection.count()

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Insert or update chunks. Returns number of records written."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal length")

        self._collection.upsert(
            ids=[_chunk_id(c) for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_name": c.document_name,
                    "page_number": c.page_number,
                    "source": c.source,
                }
                for c in chunks
            ],
        )
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return the most similar chunks for a query embedding."""
        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results: list[SearchResult] = []
        docs = response.get("documents", [[]])[0]
        metas = response.get("metadatas", [[]])[0]
        dists = response.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            results.append(
                SearchResult(
                    text=doc,
                    score=1.0 - float(dist),  # cosine similarity
                    document_name=meta.get("document_name", "unknown"),
                    page_number=int(meta.get("page_number", 0)),
                    source=meta.get("source", "Student Handbook"),
                )
            )
        return results
