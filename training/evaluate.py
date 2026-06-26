from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
)

from dataset import (
    ModerationDataset,
    get_tokenizer,
)


# ============================================================
# CONFIG
# ============================================================

MAX_LENGTH = 256
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROOT_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = ROOT_DIR / "data" / "processed" / "test.csv"

MODEL_PATH = ROOT_DIR / "model" / "best_model"

LABEL_MAP = {
    0: "ALLOW",
    1: "FLAG",
    2: "BLOCK"
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Loading model...")
print("=" * 60)

tokenizer = get_tokenizer()

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()


# ============================================================
# LOAD TEST DATA
# ============================================================

test_dataset = ModerationDataset(
    csv_path=TEST_PATH,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_df = pd.read_csv(TEST_PATH)

print(f"Test Samples: {len(test_dataset):,}")


# ============================================================
# INFERENCE
# ============================================================

all_preds = []
all_labels = []

print("\nRunning inference...")

with torch.no_grad():

    for batch in tqdm(
        test_loader,
        desc="Evaluating"
    ):

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        preds = torch.argmax(
            outputs.logits,
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            batch["labels"].numpy()
        )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    all_labels,
    all_preds,
    target_names=[
        "ALLOW",
        "FLAG",
        "BLOCK"
    ],
    digits=4,
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    all_labels,
    all_preds
)

print(cm)

print("\nRows = Actual")
print("Cols = Predicted")

print("""
            ALLOW  FLAG  BLOCK
ALLOW
FLAG
BLOCK
""")


# ============================================================
# ERROR ANALYSIS
# ============================================================

results_df = test_df.copy()

results_df["true_label"] = [
    LABEL_MAP[x]
    for x in all_labels
]

results_df["pred_label"] = [
    LABEL_MAP[x]
    for x in all_preds
]


def show_errors(
    actual,
    predicted,
    n=5
):
    """
    Print sample mistakes.
    """

    errors = results_df[
        (results_df["true_label"] == actual)
        &
        (results_df["pred_label"] == predicted)
    ]

    print("\n" + "=" * 60)
    print(
        f"{actual} → {predicted}"
    )
    print(
        f"Count: {len(errors)}"
    )
    print("=" * 60)

    samples = errors.head(n)

    if len(samples) == 0:
        print("No examples found.")
        return

    for idx, row in samples.iterrows():

        print("\n--- Example ---")

        text = (
            str(row["comment_text"])
            .replace("\n", " ")
        )

        print(
            text[:500]
        )

        print(
            f"\nActual: {actual}"
        )

        print(
            f"Predicted: {predicted}"
        )


# ============================================================
# MOST IMPORTANT ERRORS
# ============================================================

print("\n")
print("=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

# Dangerous misses
show_errors(
    actual="BLOCK",
    predicted="ALLOW"
)

# Toxic missed
show_errors(
    actual="FLAG",
    predicted="ALLOW"
)

# False positive
show_errors(
    actual="ALLOW",
    predicted="BLOCK"
)

print("\nEvaluation complete.")