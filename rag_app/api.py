from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .handbook import HandbookKnowledgeBase

app = FastAPI(title="Student Handbook Assistant", version="1.0.0")


@app.on_event("startup")
def startup_event() -> None:
    knowledge_base = HandbookKnowledgeBase(data_dir="data")
    default_handbook = knowledge_base.load_default_handbook()
    knowledge_base.index_documents(default_handbook)
    app.state.index = knowledge_base


@app.post("/ask")
async def ask_question(request: Request) -> dict[str, str]:
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON") from None

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON")

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=400, detail="Question must be a non-empty string.")

    try:
        answer, source = app.state.index.answer_question(question)
        return {"answer": answer, "source": source}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.exception_handler(HTTPException)
def handle_http_exception(_: object, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
