"""Embedding generation with Sentence Transformers.

The model is loaded lazily and cached for the process lifetime so the
~90MB download/load only happens once. Documents and questions are
embedded with the same model, guaranteeing compatible dimensions.
"""

from __future__ import annotations

import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class Embedder:
    """Thin wrapper around SentenceTransformer with lazy loading."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        """Return the loaded model, loading it on first access."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding vector size (384 for all-MiniLM-L6-v2)."""
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks."""
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user question (same model => same dimensions)."""
        vector = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector[0].astype(np.float32).tolist()
