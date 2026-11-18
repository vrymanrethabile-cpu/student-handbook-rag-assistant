"""Tests for chunking, embeddings, vector storage and retrieval (Phases 3-5).

Uses the real (local, free) Sentence Transformer and a temporary ChromaDB
directory — no paid API calls involved.
"""

import pytest

from app.ingestion.embedder import Embedder
from app.ingestion.pdf_loader import PageRecord
from app.ingestion.text_splitter import Chunk, split_pages
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture
def sample_pages():
    return [
        PageRecord(
            document_name="test_handbook.pdf",
            page_number=1,
            text=(
                "Attendance Policy: Students must attend at least 80% of all classes. "
                "In cases of illness, students must provide a medical certificate within three days. "
                "The attendance policy is strictly enforced by the faculty."
            ),
        ),
        PageRecord(
            document_name="test_handbook.pdf",
            page_number=2,
            text=(
                "Library Services: Students may borrow up to five books for a maximum "
                "period of 21 days. Late returns incur a fee per day. The library is "
                "open from 8am to 8pm on weekdays."
            ),
        ),
        PageRecord(
            document_name="other_handbook.pdf",
            page_number=5,
            text=(
                "Examinations: Students must pass all major assessments to qualify "
                "for the final exam. Deferred exams require an approved application "
                "submitted at least two weeks in advance."
            ),
        ),
    ]


@pytest.fixture
def chunks(sample_pages):
    return split_pages(sample_pages, chunk_size=400, chunk_overlap=80)


@pytest.fixture
def store(tmp_path, chunks, embedder):
    vs = VectorStore(db_path=tmp_path / "chroma_test", collection_name="test")
    embeddings = embedder.embed_documents([c.text for c in chunks])
    vs.upsert_chunks(chunks, embeddings)
    return vs


# --- Chunking ---

def test_split_pages_produces_chunks_with_metadata(chunks):
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.text.strip()
        assert chunk.document_name.endswith(".pdf")
        assert chunk.page_number >= 1
        assert chunk.source == "Student Handbook"


def test_chunks_never_mix_pages(chunks):
    # Each chunk must come from exactly one page
    for chunk in chunks:
        assert chunk.page_number in {1, 2, 5}


# --- Embeddings ---

def test_embedding_dimensions_match(embedder, chunks):
    doc_vecs = embedder.embed_documents([c.text for c in chunks])
    query_vec = embedder.embed_query("What is the attendance policy?")
    assert len(doc_vecs) == len(chunks)
    assert all(len(v) == embedder.dimension for v in doc_vecs)
    assert len(query_vec) == embedder.dimension  # compatible dimensions


# --- Vector store ---

def test_vector_store_upsert_and_count(store, chunks):
    assert store.count == len(chunks)


def test_vector_store_dedup_on_reupsert(store, chunks, embedder, tmp_path):
    embeddings = embedder.embed_documents([c.text for c in chunks])
    store.upsert_chunks(chunks, embeddings)  # same chunks again
    assert store.count == len(chunks)  # no duplicates


def test_search_returns_relevant_chunk_with_metadata(store, embedder):
    query_vec = embedder.embed_query("How many books can I borrow?")
    results = store.search(query_vec, top_k=3)
    assert results
    top = results[0]
    assert "book" in top.text.lower() or "borrow" in top.text.lower()
    assert top.page_number == 2
    assert top.document_name == "test_handbook.pdf"
    assert 0.0 <= top.score <= 1.0


# --- Retriever ---

def test_retriever_finds_attendance_chunk(store, embedder):
    retriever = Retriever(store=store, embedder=embedder, top_k=3, threshold=0.2)
    result = retriever.retrieve("What is the attendance requirement?")
    assert result.has_relevant_context
    assert any("80%" in c.text or "attendance" in c.text.lower() for c in result.chunks)


def test_retriever_threshold_blocks_unrelated(store, embedder):
    retriever = Retriever(store=store, embedder=embedder, top_k=3, threshold=0.99)
    result = retriever.retrieve("What is the attendance requirement?")
    assert not result.has_relevant_context
    assert result.chunks == []


def test_retriever_empty_question(store, embedder):
    retriever = Retriever(store=store, embedder=embedder)
    result = retriever.retrieve("   ")
    assert not result.has_relevant_context
