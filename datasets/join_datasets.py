import pandas as pd

def analyze_dataset(df, name):
    """
    Prints the total number of entries and label distribution in the dataset.
    """
    print(f"\n--- Analysis for: {name} ---")
    print(f"Total entries: {len(df)}")
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().sort_index()
        print("Label counts:")
        print(f"  Label 0: {label_counts.get(0, 0)} entries")
        print(f"  Label 1: {label_counts.get(1, 0)} entries")
        for label in label_counts.index:
            if label not in [0, 1]:
                print(f"  Label {label}: {label_counts[label]} entries (other labels)")
    else:
        print("No 'label' column found.")

# Load datasets
lardelli = pd.read_csv("lardelli_synthetically_extended.csv")
mgente = pd.read_csv("mgente_transformed.csv")

# Filter by label
lardelli_biased = lardelli[lardelli["label"] == 1]
lardelli_neutral = lardelli[lardelli["label"] == 0]
mgente_biased = mgente[mgente["label"] == 1]
mgente_neutral = mgente[mgente["label"] == 0]

# Sample
sampled_lardelli_biased = lardelli_biased.sample(n=1300, random_state=10)
sampled_mgente_biased = mgente_biased.sample(n=1300, random_state=10)
sampled_lardelli_neutral = lardelli_neutral.sample(n=1300, random_state=10)
sampled_mgente_neutral = mgente_neutral.sample(n=1300, random_state=10)

# Combine, shuffle, save
final_df = pd.concat([
    sampled_lardelli_biased,
    sampled_mgente_biased,
    sampled_lardelli_neutral,
    sampled_mgente_neutral
], ignore_index=True)

final_df = final_df.sample(frac=1, random_state=10).reset_index(drop=True)
final_df.to_csv("mgente_lardelli_equal.csv", index=False)

print("Saved as mgente_lardelli_equal.csv")

# Run summary
analyze_dataset(lardelli, "lardelli_and_gpt_data.csv")
analyze_dataset(mgente, "mgente_transformed.csv")
analyze_dataset(final_df, "mgente_lardelli_equal.csv (final)")
