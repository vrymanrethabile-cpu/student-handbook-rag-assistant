"""Text chunking with overlap using LangChain's RecursiveCharacterTextSplitter.

Each chunk keeps metadata (source, page number, document name) so the
API can cite the exact handbook page an answer came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.pdf_loader import PageRecord


@dataclass
class Chunk:
    """A searchable piece of handbook text with provenance metadata."""

    text: str
    document_name: str
    page_number: int
    source: str = "Student Handbook"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise for ChromaDB storage."""
        return {
            "text": self.text,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "source": self.source,
            "metadata": self.metadata,
        }


def split_pages(
    pages: list[PageRecord],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Split page records into overlapping chunks.

    Splitting is done per page so a chunk never mixes content from two
    different pages — page attribution stays exact.

    Args:
        pages: PageRecord list from pdf_loader.
        chunk_size: Target chunk size in characters (~200-250 tokens).
        chunk_overlap: Overlap between consecutive chunks so sentences
            split at boundaries are still retrievable.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[Chunk] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            # Skip near-empty fragments (cover pages, dividers) — they
            # contain no answers but can outrank real content in search.
            if len(piece) < 100:
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    document_name=page.document_name,
                    page_number=page.page_number,
                    source=page.source,
                    metadata={
                        "document_name": page.document_name,
                        "page_number": page.page_number,
                        "source": page.source,
                    },
                )
            )
    return chunks
