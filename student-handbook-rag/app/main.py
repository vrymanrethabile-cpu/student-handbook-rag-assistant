"""FastAPI application entry point.

Exposes:
    GET  /health - liveness check
    POST /ask    - answer a student question from the handbook
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.generation.answer_generator import AnswerGenerator
from app.retrieval.retriever import Retriever
from app.schemas.request_models import AskRequest, AskResponse

app = FastAPI(
    title="Student Handbook RAG Assistant",
    version="0.1.0",
    description="Answers student questions using only the Student Handbook PDF.",
)


def _get_pipeline(request: Request) -> tuple[Retriever, AnswerGenerator]:
    """Lazily build the pipeline on first request.

    Lazy init means the app works regardless of whether the ASGI
    lifespan/startup events have run (e.g. TestClient without a
    context manager). The handbook is NOT re-indexed here — embeddings
    live in the persistent ChromaDB collection created by
    scripts/ingest_handbook.py.
    """
    if not hasattr(request.app.state, "retriever"):
        request.app.state.retriever = Retriever()
        request.app.state.generator = AnswerGenerator()
    return request.app.state.retriever, request.app.state.generator


@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness probe used by tests and deployment platforms."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: Request, payload: AskRequest) -> AskResponse:
    """Answer a student question using retrieved handbook context.

    Request:  {"question": "What is the attendance requirement?"}
    Response: {"answer": "...", "source": "handbook - Page 5"}
    """
    question = payload.question.strip()
    retriever, generator = _get_pipeline(request)

    try:
        retrieval = retriever.retrieve(question)
    except Exception:
        return JSONResponse(
            status_code=503,
            content={
                "answer": "The knowledge base is unavailable. "
                "Please run scripts/ingest_handbook.py first.",
                "source": None,
            },
        )

    if not retrieval.has_relevant_context:
        return AskResponse(
            answer="I could not find that information in the available knowledge base.",
            source=None,
        )

    try:
        answer = generator.generate(question, retrieval.chunks)
    except Exception:
        answer = "I could not find that information in the available knowledge base."

    # If generation fell back, do not cite a source for an unsupported answer
    if "could not find" in answer.lower():
        return AskResponse(answer=answer, source=None)

    return AskResponse(answer=answer, source=generator.source_for(retrieval.chunks, answer))
