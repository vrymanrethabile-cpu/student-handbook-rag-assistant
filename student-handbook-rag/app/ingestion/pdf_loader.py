"""PDF loading and text extraction using PyMuPDF (fitz).

Each page is extracted separately so the original PDF page number is
preserved in the metadata — this is what lets the API tell a student
exactly which page an answer came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageRecord:
    """Text extracted from a single PDF page, with provenance metadata."""

    document_name: str  # e.g. "fullstack-web-dev-handbook.pdf"
    page_number: int    # 1-indexed original PDF page number
    text: str           # cleaned page text
    source: str = "Student Handbook"


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove common PDF extraction artefacts.

    - Collapses runs of spaces/tabs into a single space.
    - Collapses 3+ newlines into a paragraph break (2 newlines).
    - Strips trailing spaces on each line.
    """
    # Normalise line endings and strip trailing whitespace per line
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    text = "\n".join(lines)
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse excessive blank lines into a single paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pages(pdf_path: str | Path) -> list[PageRecord]:
    """Extract text from every page of a PDF.

    Args:
        pdf_path: Path to the handbook PDF.

    Returns:
        List of PageRecord, one per non-empty page, in page order.

    Raises:
        FileNotFoundError: If the PDF does not exist, with a message
            telling the user where to place the file.
        ValueError: If the file exists but cannot be opened as a PDF.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Handbook PDF not found at: {path}\n"
            f"Place the Student Handbook PDF in the 'data/' folder "
            f"(expected location: {path}) and try again."
        )

    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # fitz raises various exceptions for bad files
        raise ValueError(f"Could not open '{path}' as a PDF: {exc}") from exc

    records: list[PageRecord] = []
    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            raw_text = page.get_text("text")
            text = _clean_text(raw_text)

            # Skip pages with no extractable text (blank pages, image-only
            # pages) — they cannot contain answers and would only add noise.
            if not text:
                continue

            records.append(
                PageRecord(
                    document_name=path.name,
                    page_number=page_index + 1,  # humans count from 1
                    text=text,
                )
            )
    finally:
        doc.close()

    return records


def load_all_handbooks(data_dir: str | Path) -> list[PageRecord]:
    """Load every PDF found in the data directory.

    Supports multiple handbooks: each PDF becomes part of the same
    knowledge base, distinguished by its ``document_name`` metadata.
    PDFs are processed in sorted filename order for deterministic output.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {directory}. "
            f"Create it and place the handbook PDF(s) inside."
        )

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{directory}'. "
            f"Place the Student Handbook PDF(s) there and try again."
        )

    records: list[PageRecord] = []
    for pdf_file in pdf_files:
        records.extend(load_pdf_pages(pdf_file))
    return records
