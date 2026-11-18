"""Question retrieval pipeline.

Embeds the student's question with the same Sentence Transformer used
for the handbook chunks, searches ChromaDB, and applies a similarity
threshold so unrelated questions retrieve nothing instead of random
chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import SearchResult, VectorStore


@dataclass
class RetrievalResult:
    """Outcome of a retrieval call."""

    chunks: list[SearchResult]
    has_relevant_context: bool


class Retriever:
    """Turns a question into relevant handbook chunks."""

    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ):
        self.store = store or VectorStore()
        self.embedder = embedder or Embedder()
        self.top_k = top_k if top_k is not None else settings.retrieval_top_k
        self.threshold = (
            threshold if threshold is not None else settings.similarity_threshold
        )

    def retrieve(self, question: str) -> RetrievalResult:
        """Embed the question, search ChromaDB, filter by threshold."""
        if not question or not question.strip():
            return RetrievalResult(chunks=[], has_relevant_context=False)

        query_embedding = self.embedder.embed_query(question)
        results = self.store.search(query_embedding, top_k=self.top_k)

        relevant = [r for r in results if r.score >= self.threshold]
        return RetrievalResult(
            chunks=relevant,
            has_relevant_context=len(relevant) > 0,
        )
