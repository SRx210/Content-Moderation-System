"""
Label Engineering for Content Moderation

Input:
    data/raw/train.csv

Output:
    data/processed/labeled.csv

Moderation Labels:
    BLOCK -> severe_toxic OR threat OR identity_hate
    FLAG  -> toxic OR obscene OR insult
    ALLOW -> everything else
"""

from pathlib import Path
import pandas as pd



# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT_DIR / "data" / "raw" / "train.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "labeled.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load Data
print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print(f"Dataset Shape: {df.shape}")


# Label Engineering
print("\nApplying moderation label logic...")

df["moderation_label"] = "ALLOW"

# BLOCK (highest priority)
block_mask = (
    (df["severe_toxic"] == 1)
    | (df["threat"] == 1)
    | (df["identity_hate"] == 1)
)

df.loc[block_mask, "moderation_label"] = "BLOCK"

# FLAG (only if not already BLOCK)
flag_mask = (
    (df["toxic"] == 1)
    | (df["obscene"] == 1)
    | (df["insult"] == 1)
)

df.loc[
    (~block_mask) & flag_mask,
    "moderation_label"
] = "FLAG"


# Distribution
print("\n" + "=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)

label_counts = df["moderation_label"].value_counts()

for label in ["ALLOW", "FLAG", "BLOCK"]:
    count = label_counts.get(label, 0)
    percentage = (count / len(df)) * 100

    print(
        f"{label:<6} : "
        f"{count:>7,} samples "
        f"({percentage:.2f}%)"
    )


# Sample Rows
print("\n" + "=" * 60)
print("SAMPLE ROWS")
print("=" * 60)

for label in ["ALLOW", "FLAG", "BLOCK"]:

    print(f"\n--- {label} Samples ---")

    samples = (
        df[df["moderation_label"] == label]
        [["comment_text", "moderation_label"]]
        .head(3)
    )

    for idx, row in samples.iterrows():
        text = str(row["comment_text"]).replace("\n", " ")
        print(f"\n[{idx}]")
        print(text[:250])
        print(f"Label: {row['moderation_label']}")


# Save
df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 60)
print("Saved processed dataset")
print("=" * 60)
print(f"Output: {OUTPUT_PATH}")
print(f"Total Rows Saved: {len(df):,}")