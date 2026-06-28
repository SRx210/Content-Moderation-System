# inference/postprocessor.py

import numpy as np


LABEL_MAP = {0: "ALLOW", 1: "FLAG", 2: "BLOCK"}


def postprocess(logits: np.ndarray) -> dict:
    probs      = softmax(logits)
    pred_idx   = int(np.argmax(probs))
    label      = LABEL_MAP[pred_idx]
    confidence = float(probs[pred_idx])

    scores = {
        "ALLOW": float(probs[0]),
        "FLAG":  float(probs[1]),
        "BLOCK": float(probs[2]),
    }

    return {
        "label":      label,
        "confidence": confidence,
        "scores":     scores,
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()
