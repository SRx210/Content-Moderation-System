# inference/onnx_session.py

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class ONNXInferenceSession:
    def __init__(self, onnx_model_path: str, tokenizer_path: str, max_length: int = 256):
        self.max_length = max_length
        self.label_map  = {0: "ALLOW", 1: "FLAG", 2: "BLOCK"}

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load ONNX session
        session_options                            = ort.SessionOptions()
        session_options.graph_optimization_level  = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads       = 4

        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    def predict(self, text: str) -> dict:
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        encoding["input_ids"]      = encoding["input_ids"].astype(np.int64)
        encoding["attention_mask"] = encoding["attention_mask"].astype(np.int64)

        # Run ONNX inference
        logits = self.session.run(
            ["logits"],
            {
                "input_ids":      encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
            },
        )[0]

        # Softmax to get probabilities
        probs      = self._softmax(logits[0])
        pred_idx   = int(np.argmax(probs))
        label      = self.label_map[pred_idx]
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

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()
