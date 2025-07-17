import pandas as pd

SEED = 10

def analyze_dataset(df, name):
    """
    Prints the total number of entries and label distribution in the dataset,
    including both counts and percentages.
    """
    print(f"\n--- Analysis for: {name} ---")
    print(f"Total entries: {len(df)}")
    
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().sort_index()
        total = len(df)
        
        print("Label distribution:")
        for label in label_counts.index:
            count = label_counts[label]
            percentage = (count / total) * 100
            if label in [0, 1]:
                print(f"  Label {label}: {count} entries ({percentage:.1f}%)")
            else:
                print(f"  Label {label}: {count} entries ({percentage:.1f}%) (other labels)")
    else:
        print("No 'label' column found.")


# Load datasets
lardelli = pd.read_csv("lardelli_synthetically_extended.csv")
mgente = pd.read_csv("mgente_transformed_fair.csv")
deu = pd.read_csv("deu_processed_sampled.csv")

# Sample data
samples = {
    "lardelli": {"biased_n": 600, "neutral_n": 600},
    "mgente": {"biased_n": 750, "neutral_n": 750},
    "deu": {"biased_n": 0, "neutral_n": 500}
}

def sample_data(df, biased_n, neutral_n):
    biased = df[df["label"] == 1].sample(n=biased_n, random_state=SEED)
    neutral = df[df["label"] == 0].sample(n=neutral_n, random_state=SEED)
    return pd.concat([biased, neutral], ignore_index=True)

sampled_lardelli = sample_data(lardelli, **samples["lardelli"])
sampled_mgente = sample_data(mgente, **samples["mgente"])
sampled_deu = sample_data(deu, **samples["deu"])

# Combine, shuffle
final_df = pd.concat([sampled_lardelli, sampled_mgente, sampled_deu], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Check for NaNs
missing = final_df.isna().sum()
if missing.any():
    print("\nWarning: Missing values found before saving:")
    print(missing)
else:
    print("\nNo missing values found.")

name = "dataset_fair.csv"

final_df.to_csv(name, index=False)
print("Saved as ",name)

analyze_dataset(final_df, name)
