# api/main.py

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.routers import health, moderation
from inference.onnx_session import ONNXInferenceSession


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ONNX session once on startup
    app.state.onnx_session = ONNXInferenceSession(
        onnx_model_path=settings.onnx_model_path,
        tokenizer_path=settings.tokenizer_path,
        max_length=settings.max_length,
    )
    print("ONNX session loaded.")
    yield
    # Cleanup on shutdown
    del app.state.onnx_session
    print("ONNX session released.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(moderation.router, tags=["Moderation"])