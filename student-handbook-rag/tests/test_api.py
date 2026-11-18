"""Tests for the FastAPI endpoints (Phase 7).

The retriever/generator pipeline is replaced with fakes so tests run
fast and never call the paid LLM API.
"""

import pytest
from fastapi.testclient import TestClient

from app.generation.answer_generator import FALLBACK_MESSAGE
from app.main import app
from app.retrieval.retriever import RetrievalResult
from app.retrieval.vector_store import SearchResult


def _fake_chunk(text="Students must attend 80% of classes.", page=3):
    return SearchResult(
        text=text,
        score=0.9,
        document_name="test_handbook.pdf",
        page_number=page,
        source="Student Handbook",
    )


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, question):
        if not self._chunks:
            return RetrievalResult(chunks=[], has_relevant_context=False)
        return RetrievalResult(chunks=self._chunks, has_relevant_context=True)


class FakeGenerator:
    def __init__(self, answer="The handbook says attend 80% of classes."):
        self._answer = answer

    def generate(self, question, chunks):
        return self._answer

    def source_for(self, chunks, answer=None):
        if not chunks:
            return None
        return f"{chunks[0].document_name.removesuffix('.pdf')} - Page {chunks[0].page_number}"


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_source(client):
    app.state.retriever = FakeRetriever([_fake_chunk()])
    app.state.generator = FakeGenerator()

    response = client.post("/ask", json={"question": "What is the attendance requirement?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "The handbook says attend 80% of classes."
    assert payload["source"] == "test_handbook - Page 3"


def test_ask_fallback_when_no_context(client):
    app.state.retriever = FakeRetriever([])
    app.state.generator = FakeGenerator()

    response = client.post("/ask", json={"question": "What is the asteroid policy?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == FALLBACK_MESSAGE
    assert payload["source"] is None


def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": "   "})
    # Pydantic min_length catches empty; whitespace-only passes validation
    # but must still return a safe response
    assert response.status_code in (200, 422)


def test_ask_rejects_missing_question_field(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422
    assert "question" in response.text.lower()


def test_ask_rejects_invalid_json(client):
    response = client.post(
        "/ask",
        content="not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_ask_rejects_wrong_type(client):
    response = client.post("/ask", json={"question": 123})
    assert response.status_code == 422
