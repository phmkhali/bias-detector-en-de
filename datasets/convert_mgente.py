import pandas as pd

# Load your mGeNTE dataset
df = pd.read_csv("mgente_en-de.csv", sep=';')  # adjust sep if needed

# Prepare final records
records = []

for idx, row in df.iterrows():
    src = row['SRC']
    ref_g = row['REF-G']
    ref_n = row['REF-N']
    set_type = row['SET']

    if pd.isna(ref_g) or pd.isna(ref_n):
        continue  # skip incomplete rows

    if set_type == 'Set-G':
        # English has gender clue
        # REF-G = faithful (0), REF-N = dropped info (1)
        records.append({'source_text': src, 'translation': ref_g, 'label': 0})
        records.append({'source_text': src, 'translation': ref_n, 'label': 1})

    elif set_type == 'Set-N':
        # English is gender ambiguous
        # REF-G = inserts gender (1), REF-N = faithful (0)
        records.append({'source_text': src, 'translation': ref_g, 'label': 1})
        records.append({'source_text': src, 'translation': ref_n, 'label': 0})

# Create and save the final DataFrame
final_df = pd.DataFrame(records)
final_df.to_csv("mgente_transformed.csv", index=False)
