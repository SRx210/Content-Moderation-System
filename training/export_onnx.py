# training/export_onnx.py

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import onnxruntime as ort


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR       = Path(__file__).resolve().parent.parent
BEST_MODEL_DIR = ROOT_DIR / "model" / "best_model"
ONNX_PATH      = ROOT_DIR / "model" / "distilbert_moderation.onnx"

MAX_LENGTH     = 256
DEVICE         = torch.device("cpu")  # ONNX export must be on CPU


# ============================================================
# LOAD PYTORCH MODEL
# ============================================================

print("=" * 60)
print("Loading PyTorch model from checkpoint...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR, num_labels=3)
model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# CREATE DUMMY INPUT
# ============================================================

dummy_text = "This is a sample comment for ONNX export."

encoding = tokenizer(
    dummy_text,
    max_length=MAX_LENGTH,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

dummy_input_ids      = encoding["input_ids"].to(DEVICE)
dummy_attention_mask = encoding["attention_mask"].to(DEVICE)


# ============================================================
# EXPORT TO ONNX
# ============================================================

print("\nExporting model to ONNX...")

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    str(ONNX_PATH),
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids":      {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits":         {0: "batch_size"},
    },
    opset_version=14,
    do_constant_folding=True,
)

print(f"ONNX model saved to: {ONNX_PATH}")


# ============================================================
# VALIDATE: COMPARE PYTORCH VS ONNX OUTPUTS
# ============================================================

print("\n" + "=" * 60)
print("Validating ONNX output against PyTorch output...")
print("=" * 60)

# PyTorch inference
with torch.no_grad():
    pytorch_logits = model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
    ).logits.numpy()

# ONNX inference
ort_session = ort.InferenceSession(str(ONNX_PATH))

onnx_logits = ort_session.run(
    ["logits"],
    {
        "input_ids":      dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy(),
    },
)[0]

# Compare
max_diff = np.max(np.abs(pytorch_logits - onnx_logits))
print(f"Max logit difference (PyTorch vs ONNX): {max_diff:.6f}")

if max_diff < 1e-4:
    print("Validation PASSED — outputs match.")
else:
    print("WARNING — outputs differ more than expected. Check export settings.")


# ============================================================
# MODEL SIZE COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("Model Size Comparison")
print("=" * 60)

pytorch_size_mb = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
) / 1e6

onnx_size_mb = ONNX_PATH.stat().st_size / 1e6

print(f"PyTorch model size : {pytorch_size_mb:.1f} MB")
print(f"ONNX model size    : {onnx_size_mb:.1f} MB")


# ============================================================
# LABEL MAP
# ============================================================

print("\n" + "=" * 60)
print("Sample Prediction Check")
print("=" * 60)

LABEL_MAP = {0: "ALLOW", 1: "FLAG", 2: "BLOCK"}

pytorch_pred = LABEL_MAP[int(np.argmax(pytorch_logits))]
onnx_pred    = LABEL_MAP[int(np.argmax(onnx_logits))]

print(f"Input text     : {dummy_text}")
print(f"PyTorch pred   : {pytorch_pred}")
print(f"ONNX pred      : {onnx_pred}")

print("\n" + "=" * 60)
print("ONNX Export Complete")
print("=" * 60)