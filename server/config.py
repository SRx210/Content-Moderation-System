# api/config.py

from pathlib import Path
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str       = "Content Moderation API"
    app_version: str    = "1.0.0"
    debug: bool         = False

    onnx_model_path: str    = str(ROOT_DIR / "model" / "distilbert_moderation_quantized.onnx")
    tokenizer_path: str     = str(ROOT_DIR / "model" / "best_model")
    max_length: int         = 256

    log_dir: str = str(ROOT_DIR / "logs")

    class Config:
        env_file = ".env"


settings = Settings()
