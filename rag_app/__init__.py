"""RAG application package for the student handbook."""

from .api import app
from .handbook import HandbookKnowledgeBase

__all__ = ["HandbookKnowledgeBase", "app"]
