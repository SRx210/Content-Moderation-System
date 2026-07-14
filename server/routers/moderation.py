# api/routers/moderation.py

import time

from fastapi import APIRouter, Request

from server.schemas.moderation import ModerationRequest, ModerationResponse
from server.services.preprocessor import preprocess
from server.services.logger import log_event

router = APIRouter()


@router.post("/moderate", response_model=ModerationResponse)
async def moderate(request: ModerationRequest, req: Request):
    start = time.perf_counter()

    # Preprocess text
    clean_text = preprocess(request.text)

    # Run inference via ONNX session (loaded in app state)
    session = req.app.state.onnx_session
    result  = session.predict(clean_text)

    latency_ms = (time.perf_counter() - start) * 1000

    # Log the event
    log_event(
        text=clean_text,
        label=result["label"],
        confidence=result["confidence"],
        latency_ms=latency_ms,
    )

    return ModerationResponse(
        text=clean_text,
        label=result["label"],
        confidence=result["confidence"],
        scores=result["scores"],
        latency_ms=round(latency_ms, 2),
    )
