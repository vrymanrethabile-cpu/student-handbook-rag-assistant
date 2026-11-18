"""LLM answer generation grounded in retrieved handbook context.

The prompt contains ONLY the retrieved chunks — the model is instructed
to answer exclusively from that context and to use the exact fallback
message when the context does not contain the answer. Provider and
credentials come from environment variables via app.config; nothing is
hardcoded.
"""

from __future__ import annotations

import re

import httpx

from app.config import settings
from app.retrieval.vector_store import SearchResult

FALLBACK_MESSAGE = "I could not find that information in the available knowledge base."
NOT_CONFIGURED_MESSAGE = (
    "The LLM provider is not configured. Set LLM_PROVIDER and the matching "
    "API key (e.g. GROQ_API_KEY) in your .env file. See .env.example."
)

_SYSTEM_PROMPT = (
    "You are a Student Handbook Assistant. Answer the student's question "
    "using ONLY the handbook excerpts provided in the context. "
    "Rules:\n"
    "- Base your answer strictly on the context. Do not use outside knowledge.\n"
    "- Do not invent facts, policies, or page numbers.\n"
    "- Be clear and concise (2-4 sentences).\n"
    "- Do not include excerpt numbers, brackets, or citation markers "
    "like [Excerpt 1] in your answer.\n"
    "- If the context does not contain the answer, reply with exactly: "
    f"{FALLBACK_MESSAGE}"
)


def _content_words(text: str) -> set[str]:
    """Extract normalised content words for overlap matching.

    Lowercases, strips punctuation, keeps words >= 4 chars, and adds a
    naive singular form so 'lectures' matches 'lecture' and
    'Classroom.' matches 'classroom'.
    """
    words: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", text.lower()):
        if len(raw) < 4:
            continue
        words.add(raw)
        if raw.endswith("s") and len(raw) > 4:
            words.add(raw[:-1])  # crude singular: lectures -> lecture
    return words


def _build_prompt(question: str, chunks: list[SearchResult]) -> str:
    """Assemble the user prompt with numbered context excerpts."""
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Excerpt {i} — {chunk.document_name}, Page {chunk.page_number}]\n{chunk.text}"
        )
    context = "\n\n".join(context_parts)
    return (
        f"Context from the Student Handbook:\n\n{context}\n\n"
        f"Student question: {question}\n\nAnswer:"
    )


class AnswerGenerator:
    """Generates grounded answers via a configurable LLM provider."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or settings.llm_provider).lower()
        self.model = model or settings.llm_model

    # ------------------------------------------------------------------
    # Provider call
    # ------------------------------------------------------------------
    def _call_groq(self, prompt: str) -> str:
        """Call Groq's OpenAI-compatible chat completions endpoint."""
        if not settings.groq_api_key:
            return NOT_CONFIGURED_MESSAGE
        url = f"{settings.groq_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,  # gpt-oss models are reasoning models
        }
        headers = {"Authorization": f"Bearer {settings.groq_api_key}"}
        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _call_llm(self, prompt: str) -> str:
        """Dispatch to the configured provider."""
        if self.provider == "groq":
            return self._call_groq(prompt)
        # Other providers can be added here (openai, anthropic, google, ollama)
        return NOT_CONFIGURED_MESSAGE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, question: str, chunks: list[SearchResult]) -> str:
        """Answer a question from retrieved chunks, or return the fallback."""
        if not chunks:
            return FALLBACK_MESSAGE

        try:
            answer = self._call_llm(_build_prompt(question, chunks)).strip()
        except Exception:
            # Never leak provider errors to students; fall back safely.
            return FALLBACK_MESSAGE

        if not answer or FALLBACK_MESSAGE.lower() in answer.lower():
            return FALLBACK_MESSAGE
        return answer

    def source_for(
        self, chunks: list[SearchResult], answer: str | None = None
    ) -> str | None:
        """Format the citation for the chunk the answer came from.

        When the generated answer is provided, the chunk sharing the most
        content words with it is cited — the top-ranked chunk is not
        always the one the LLM actually used. Falls back to the
        highest-ranked chunk when no overlap is found.
        """
        if not chunks:
            return None

        chosen = chunks[0]
        if answer:
            answer_words = _content_words(answer)
            best_overlap = -1
            for chunk in chunks:
                overlap = len(answer_words & _content_words(chunk.text))
                if overlap > best_overlap:
                    best_overlap = overlap
                    chosen = chunk
            if best_overlap <= 0:
                chosen = chunks[0]

        doc = chosen.document_name.removesuffix(".pdf")
        return f"{doc} - Page {chosen.page_number}"
