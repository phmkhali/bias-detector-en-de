import pandas as pd

"""
This script reads the mGeNTE dataset from a CSV file with columns SRC, REF-G, REF-N, and SET.
It processes the rows based on the set type to create labeled examples of English-German sentence pairs.
If set type is 'Set-G', it adds two entries labeled 0.
If set type is 'Set-N', it adds one entry labeled 1 and one labeled 0.
Finally, it saves the transformed data to a new CSV and prints label statistics and a sample.
"""

df = pd.read_csv("mgente_en-de.csv", sep=';')

records = []

for idx, row in df.iterrows():
    src = row['SRC']
    ref_g = row['REF-G']
    ref_n = row['REF-N']
    set_type = row['SET']

    if pd.isna(ref_g) or pd.isna(ref_n):
        continue

    if set_type == 'Set-G':
        # english has gender clue
        # ref-g = faithful (0), ref-n = dropped info (1)
        records.append({'english': src, 'german': ref_g, 'label': 0})
        records.append({'english': src, 'german': ref_n, 'label': 0})
    elif set_type == 'Set-N':
        # english is gender ambiguous
        # ref-g = inserts gender (1), ref-n = faithful (0)
        records.append({'english': src, 'german': ref_g, 'label': 1})
        records.append({'english': src, 'german': ref_n, 'label': 0})

final_df = pd.DataFrame(records)
final_df.to_csv("mgente_final.csv", index=False)

print("\nlabel distribution in transformed dataset:")
label_counts = final_df['label'].value_counts().sort_index()
total = len(final_df)
for label in label_counts.index:
    count = label_counts[label]
    percentage = (count / total) * 100
    print(f"  label {label}: {count} entries ({percentage:.1f}%)")

print(f"\ntotal entries: {total}")
print(f"\nsample of the transformed dataset:")
print(final_df.head())
