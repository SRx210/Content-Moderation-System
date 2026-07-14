# api/schemas/moderation.py

from pydantic import BaseModel, Field


class ModerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to moderate")


class ModerationResponse(BaseModel):
    text: str
    label: str                  # ALLOW / FLAG / BLOCK
    confidence: float           # Confidence of predicted label
    scores: dict[str, float]    # Scores for all three classes
    latency_ms: float           # Inference latency in milliseconds
