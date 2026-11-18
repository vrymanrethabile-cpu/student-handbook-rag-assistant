"""Tests for PDF loading and text extraction (Phase 2).

Tests generate tiny PDFs on the fly with PyMuPDF, so no binary
fixtures are needed. Real-handbook loading is verified separately
by running the ingestion script.
"""

from pathlib import Path

import fitz
import pytest

from app.ingestion.pdf_loader import (
    PageRecord,
    load_all_handbooks,
    load_pdf_pages,
)


def _make_pdf(path: Path, pages: list[str]) -> Path:
    """Create a minimal PDF with one text line per page."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:  # empty string -> leave page blank
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def sample_pdf(tmp_path):
    return _make_pdf(
        tmp_path / "sample_handbook.pdf",
        [
            "Page one attendance policy text.",
            "",  # blank page must be handled
            "Page three library rules text.",
        ],
    )


def test_load_pdf_pages_extracts_text(sample_pdf):
    records = load_pdf_pages(sample_pdf)
    assert len(records) == 2  # blank page skipped
    assert "attendance policy" in records[0].text
    assert "library rules" in records[1].text


def test_page_numbers_are_preserved(sample_pdf):
    records = load_pdf_pages(sample_pdf)
    # Original PDF pages were 1 and 3 (page 2 was blank)
    assert [r.page_number for r in records] == [1, 3]


def test_metadata_is_attached(sample_pdf):
    records = load_pdf_pages(sample_pdf)
    for record in records:
        assert isinstance(record, PageRecord)
        assert record.document_name == "sample_handbook.pdf"
        assert record.source == "Student Handbook"
        assert record.page_number >= 1


def test_missing_pdf_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_pdf_pages(missing)
    message = str(exc_info.value)
    assert "data/" in message or "not found" in message.lower()


def test_invalid_file_raises_value_error(tmp_path):
    bad = tmp_path / "not_a_pdf.pdf"
    bad.write_text("this is not a pdf", encoding="utf-8")
    with pytest.raises(ValueError, match="PDF"):
        load_pdf_pages(bad)


def test_load_all_handbooks_loads_multiple_pdfs(tmp_path):
    _make_pdf(tmp_path / "a_handbook.pdf", ["Alpha handbook content."])
    _make_pdf(tmp_path / "b_handbook.pdf", ["Beta handbook content."])

    records = load_all_handbooks(tmp_path)

    assert len(records) == 2
    names = {r.document_name for r in records}
    assert names == {"a_handbook.pdf", "b_handbook.pdf"}


def test_load_all_handbooks_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No PDF"):
        load_all_handbooks(tmp_path)
