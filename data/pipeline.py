"""
Data Splitting Pipeline

Input:
    data/processed/labeled.csv

Output:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv

Split:
    Train: 80%
    Validation: 10%
    Test: 10%

Uses stratified sampling based on moderation_label.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT_DIR / "data" / "processed" / "labeled.csv"

TRAIN_PATH = ROOT_DIR / "data" / "processed" / "train.csv"
VAL_PATH = ROOT_DIR / "data" / "processed" / "val.csv"
TEST_PATH = ROOT_DIR / "data" / "processed" / "test.csv"


# Helper Function
def print_distribution(df, split_name):
    """Print count and percentage per label."""

    print(f"\n{'=' * 60}")
    print(f"{split_name.upper()} DISTRIBUTION")
    print(f"{'=' * 60}")

    total = len(df)

    counts = df["moderation_label"].value_counts()

    for label in ["ALLOW", "FLAG", "BLOCK"]:
        count = counts.get(label, 0)
        pct = (count / total) * 100

        print(
            f"{label:<6} : "
            f"{count:>7,} samples "
            f"({pct:.2f}%)"
        )


# Load Data
print("=" * 60)
print("Loading labeled dataset...")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print(f"Dataset Shape: {df.shape}")


# Train/Test Split
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["moderation_label"]
)

# temp_df = 20%
# split temp equally into val and test

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["moderation_label"]
)


# Save Files
train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)


# Verify Split Sizes
print("\n" + "=" * 60)
print("SPLIT SIZES")
print("=" * 60)

print(f"Train: {len(train_df):,}")
print(f"Val  : {len(val_df):,}")
print(f"Test : {len(test_df):,}")

total_rows = len(train_df) + len(val_df) + len(test_df)
print(f"Total: {total_rows:,}")


# Verify Stratification
print_distribution(train_df, "Train")
print_distribution(val_df, "Validation")
print_distribution(test_df, "Test")


print("Data split completed successfully")