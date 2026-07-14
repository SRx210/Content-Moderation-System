# api/services/logger.py

import json
import logging
from datetime import datetime
from pathlib import Path

from server.config import settings

LOG_FILE = Path(settings.log_dir) / "moderation.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# File handler for structured JSON logs
file_handler = logging.FileHandler(str(LOG_FILE))
file_handler.setLevel(logging.INFO)

logger = logging.getLogger("moderation")
logger.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(file_handler)


def log_event(text: str, label: str, confidence: float, latency_ms: float) -> None:
    event = {
        "timestamp":   datetime.utcnow().isoformat(),
        "label":       label,
        "confidence":  round(confidence, 4),
        "latency_ms":  round(latency_ms, 2),
        "text_length": len(text),
    }
    logger.info(json.dumps(event))


def read_logs(limit: int = 100) -> list:
    if not LOG_FILE.exists():
        return []

    lines = LOG_FILE.read_text().strip().splitlines()
    parsed = []

    for line in reversed(lines):
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(parsed) >= limit:
            break

    return parsed
