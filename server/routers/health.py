# api/routers/health.py

from fastapi import APIRouter
from server.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status":  "ok",
        "app":     settings.app_name,
        "version": settings.app_version,
    }
