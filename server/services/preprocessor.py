# api/services/preprocessor.py

import re


def preprocess(text: str) -> str:
    # Strip leading/trailing whitespace
    text = text.strip()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove null bytes
    text = text.replace("\x00", "")

    return text.strip()
