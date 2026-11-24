"""Thumbs up or down on an answer. Stored with the question and sources so it can be mined later."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.v1.deps import Services, get_services
from app.core.config import Principal
from app.core.security import rate_limited
from app.db.models import ChatMessage, Feedback

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackIn(BaseModel):
    message_id: uuid.UUID
    rating: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackOut(BaseModel):
    feedback_id: str
    message_id: str
    rating: int


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=FeedbackOut,
    summary="Rate an answer (1 = helpful, -1 = not helpful); rating again replaces it",
)
async def give_feedback(
    body: FeedbackIn,
    principal: Annotated[Principal, Depends(rate_limited)],
    services: Annotated[Services, Depends(get_services)],
) -> FeedbackOut:
    async with services.sessions() as session:
        message = await session.get(ChatMessage, body.message_id)
        if message is None or message.principal != principal.name or message.role != "assistant":
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Answer not found")
        existing = await session.scalar(
            select(Feedback).where(Feedback.message_id == body.message_id, Feedback.principal == principal.name)
        )
        if existing is None:
            existing = Feedback(
                message_id=body.message_id, principal=principal.name, rating=body.rating, comment=body.comment
            )
            session.add(existing)
        else:
            existing.rating, existing.comment = body.rating, body.comment
        await session.commit()
        return FeedbackOut(feedback_id=str(existing.id), message_id=str(body.message_id), rating=existing.rating)
