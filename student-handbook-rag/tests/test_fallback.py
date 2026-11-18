"""Tests for the 'information not found' fallback behaviour (Phase 6).

No LLM calls are made — the provider layer is stubbed.
"""

from app.generation.answer_generator import (
    FALLBACK_MESSAGE,
    NOT_CONFIGURED_MESSAGE,
    AnswerGenerator,
)
from app.retrieval.vector_store import SearchResult


def _chunk(text="Some handbook text.", page=1):
    return SearchResult(
        text=text,
        score=0.8,
        document_name="handbook.pdf",
        page_number=page,
        source="Student Handbook",
    )


def test_fallback_when_no_chunks():
    gen = AnswerGenerator(provider="groq")
    assert gen.generate("anything?", []) == FALLBACK_MESSAGE


def test_fallback_when_llm_returns_fallback_text():
    gen = AnswerGenerator(provider="groq")
    gen._call_llm = lambda prompt: FALLBACK_MESSAGE  # stub provider
    assert gen.generate("unknown?", [_chunk()]) == FALLBACK_MESSAGE


def test_fallback_when_llm_returns_empty():
    gen = AnswerGenerator(provider="groq")
    gen._call_llm = lambda prompt: "   "
    assert gen.generate("unknown?", [_chunk()]) == FALLBACK_MESSAGE


def test_fallback_when_provider_raises():
    gen = AnswerGenerator(provider="groq")

    def boom(prompt):
        raise RuntimeError("API down")

    gen._call_llm = boom
    assert gen.generate("unknown?", [_chunk()]) == FALLBACK_MESSAGE


def test_real_answer_passes_through():
    gen = AnswerGenerator(provider="groq")
    gen._call_llm = lambda prompt: "Students must attend 80% of classes."
    answer = gen.generate("attendance?", [_chunk()])
    assert answer == "Students must attend 80% of classes."


def test_source_is_none_without_chunks():
    gen = AnswerGenerator(provider="groq")
    assert gen.source_for([]) is None


def test_source_format_includes_doc_and_page():
    gen = AnswerGenerator(provider="groq")
    source = gen.source_for([_chunk(page=7)])
    assert source == "handbook - Page 7"


def test_unconfigured_provider_explains_requirement(monkeypatch):
    # Simulate missing API key
    monkeypatch.setattr(
        "app.generation.answer_generator.settings.groq_api_key", None
    )
    gen = AnswerGenerator(provider="groq")
    result = gen._call_groq("test prompt")
    assert "not configured" in result.lower()
    assert ".env" in result
