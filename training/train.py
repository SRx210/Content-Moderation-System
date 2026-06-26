# training/train.py

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from dataset import ModerationDataset, get_tokenizer


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR       = Path(__file__).resolve().parent.parent
TRAIN_PATH     = ROOT_DIR / "data" / "processed" / "train.csv"
VAL_PATH       = ROOT_DIR / "data" / "processed" / "val.csv"
MODEL_DIR      = ROOT_DIR / "model"
BEST_MODEL_DIR = MODEL_DIR / "best_model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME    = str(BEST_MODEL_DIR)   # Resume from checkpoint
MAX_LENGTH    = 256
BATCH_SIZE    = 64                    # Increased for RTX 4060
EPOCHS        = 5                     # Additional epochs
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
BLOCK_W_SCALE = 0.75
START_EPOCH   = 3                     # Already completed epochs


# ============================================================
# DATASETS & DATALOADERS
# ============================================================

print("=" * 60)
print("Loading datasets...")
print("=" * 60)

tokenizer = get_tokenizer()

train_dataset = ModerationDataset(TRAIN_PATH, tokenizer, MAX_LENGTH)
val_dataset   = ModerationDataset(VAL_PATH,   tokenizer, MAX_LENGTH)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,   # False on Windows — avoids slowdown
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

print(f"Train Samples : {len(train_dataset):,}")
print(f"Val Samples   : {len(val_dataset):,}")


# ============================================================
# MODEL
# ============================================================

print("\nLoading model from checkpoint...")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(DEVICE)

print(f"Device     : {DEVICE}")
print(f"GPU        : {torch.cuda.get_device_name(0)}")
print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = np.array(train_dataset.labels)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels,
)

class_weights[2] *= BLOCK_W_SCALE  # Reduce BLOCK over-prediction

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

print(f"\nClass Weights (ALLOW / FLAG / BLOCK): {class_weights_tensor}")

criterion = CrossEntropyLoss(weight=class_weights_tensor)


# ============================================================
# OPTIMIZER & SCHEDULER
# ============================================================

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

print(f"\nTotal steps  : {total_steps:,}")
print(f"Warmup steps : {warmup_steps:,}")
print(f"Batch size   : {BATCH_SIZE}")


# ============================================================
# VALIDATION
# ============================================================

def evaluate():
    model.eval()

    all_preds  = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report   = classification_report(
        all_labels, all_preds,
        target_names=["ALLOW", "FLAG", "BLOCK"],
        digits=4,
    )

    return avg_loss, accuracy, macro_f1, report


# ============================================================
# TRAINING LOOP
# ============================================================

best_f1 = 0.0
history = []

total_epochs = START_EPOCH + EPOCHS

print("\n" + "=" * 60)
print(f"Resuming Training from Epoch {START_EPOCH + 1}")
print(f"Running epochs {START_EPOCH + 1} to {total_epochs}")
print("=" * 60)

for epoch in range(START_EPOCH, total_epochs):

    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch + 1} / {total_epochs}")
    print("=" * 60)

    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch in progress_bar:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = criterion(outputs.logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    train_loss = running_loss / len(train_loader)
    val_loss, val_accuracy, val_macro_f1, val_report = evaluate()

    history.append({
        "epoch"       : epoch + 1,
        "train_loss"  : train_loss,
        "val_loss"    : val_loss,
        "val_macro_f1": val_macro_f1,
    })

    print(f"\nTrain Loss   : {train_loss:.4f}")
    print(f"Val Loss     : {val_loss:.4f}")
    print(f"Val Accuracy : {val_accuracy:.4f}")
    print(f"Val Macro F1 : {val_macro_f1:.4f}")
    print(f"\nClassification Report\n{'-' * 40}\n{val_report}")

    if val_macro_f1 > best_f1:
        best_f1 = val_macro_f1
        model.save_pretrained(BEST_MODEL_DIR)
        tokenizer.save_pretrained(BEST_MODEL_DIR)
        print(f"New best model saved (Macro F1 = {best_f1:.4f})")


# ============================================================
# TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"\n{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<12} {'Macro F1'}")
print("-" * 48)
for h in history:
    print(f"{h['epoch']:<8} {h['train_loss']:<14.4f} {h['val_loss']:<12.4f} {h['val_macro_f1']:.4f}")

print(f"\nBest Validation Macro F1 : {best_f1:.4f}")
print(f"Saved Model              : {BEST_MODEL_DIR}")