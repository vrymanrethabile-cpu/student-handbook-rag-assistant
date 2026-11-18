"""Pydantic schemas for API request/response validation.

Keeping request models in one place means validation rules
(e.g. non-empty question) are defined once and reused everywhere.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Payload accepted by POST /ask."""

    question: str = Field(
        ...,
        min_length=1,
        description="The student's question about the handbook.",
        examples=["What are the bootcamp's hardware requirements?"],
    )


class AskResponse(BaseModel):
    """Payload returned by POST /ask."""

    answer: str = Field(..., description="Generated answer or fallback message.")
    source: str | None = Field(
        default=None,
        description="Handbook reference, e.g. 'Student Handbook - Page 5', or null.",
    )
