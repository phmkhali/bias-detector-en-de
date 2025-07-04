import pandas as pd

# Load the original mgente CSV (already converted from TSV)
df = pd.read_csv("mgente_en-de.csv", delimiter=";")

# Prepare the new rows: one biased, one unbiased per source sentence
rows = []
for _, row in df.iterrows():
    src = row["SRC"]
    ref_g = row["REF-G"]
    ref_n = row["REF-N"]

    # Only include rows that have all required fields
    if pd.notna(src) and pd.notna(ref_g) and pd.notna(ref_n):
        rows.append({"english": src, "german": ref_g, "label": 1})  # biased
        rows.append({"english": src, "german": ref_n, "label": 0})  # unbiased

# Create DataFrame and save
formatted_df = pd.DataFrame(rows)
formatted_df.to_csv("dataset.csv", index=False)
