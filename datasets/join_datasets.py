"""
This script loads three labeled datasets:
- lardelli_final.csv
- mgente_final.csv
- deu_final.csv

It samples a fixed number of biased and neutral entries from each set.
It then joins them into a single CSV file: dataset_equal.csv

Before saving, it checks for missing values and validates label integrity.
It also prints a summary of the final label distribution.
"""

import pandas as pd

SEED = 10

def analyze_dataset(df, name):
    """Print total entries, label distribution, and check for invalid labels."""
    total = len(df)
    print(f"\n--- Analysis for: {name} ---")
    print(f"Total entries: {total}")

    if 'label' not in df.columns:
        print("No 'label' column found.")
        return

    # Validate labels and count
    label_counts = df['label'].value_counts(dropna=False).sort_index()
    invalid_labels = df[~df['label'].isin([0, 1])]

    print("Label counts:")
    for label, count in label_counts.items():
        pct = (count / total) * 100
        if label in [0, 1]:
            print(f"  Label {label}: {count} entries ({pct:.1f}%)")
        else:
            print(f"  Label {label}: {count} entries (invalid)")

    if not invalid_labels.empty:
        print(f"\nWarning: Invalid label values found:\n{invalid_labels['label'].unique()}")
    else:
        print("All labels are valid (0 or 1).")

def sample_data(df, biased_n, neutral_n):
    """Sample specified number of biased and neutral entries from a dataset."""
    biased = df[df["label"] == 1].sample(n=biased_n, random_state=SEED)
    neutral = df[df["label"] == 0].sample(n=neutral_n, random_state=SEED)
    return pd.concat([biased, neutral], ignore_index=True)

# Load datasets
lardelli = pd.read_csv("lardelli_final.csv")
mgente = pd.read_csv("mgente_final.csv")
deu = pd.read_csv("deu_final.csv")


# Define sampling sizes per dataset
samples = {
    "lardelli": {"biased_n": 750, "neutral_n": 750},
    "mgente": {"biased_n": 750, "neutral_n": 750},
    "deu": {"biased_n": 0, "neutral_n": 250}
}

# Sample entries
sampled_lardelli = sample_data(lardelli, **samples["lardelli"])
sampled_mgente = sample_data(mgente, **samples["mgente"])
sampled_deu = sample_data(deu, **samples["deu"])

# Combine and shuffle
final_df = pd.concat([sampled_lardelli, sampled_mgente, sampled_deu], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Check for missing values
missing = final_df.isna().sum()
if missing.any():
    print("\nWarning: Missing values found before saving:")
    print(missing)
else:
    print("\nNo missing values found.")

# Save to file
output_file = "dataset.csv"
final_df.to_csv(output_file, index=False)
print("Saved as", output_file)

# Run dataset analysis
analyze_dataset(final_df, output_file)
