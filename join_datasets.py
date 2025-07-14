import pandas as pd

# Load datasets
lardelli = pd.read_csv("datasets/lardelli_and_gpt_data.csv")
wiki = pd.read_csv("datasets/wikipedia_biographies_transformed.csv")
mgente = pd.read_csv("datasets/mgente_transformed.csv")

# Filter by label
lardelli_biased = lardelli[lardelli["label"] == 1]
lardelli_neutral = lardelli[lardelli["label"] == 0]
wiki_neutral = wiki[wiki["label"] == 0]
mgente_biased = mgente[mgente["label"] == 1]
mgente_neutral = mgente[mgente["label"] == 0]

# Sample
sampled_lardelli_biased = lardelli_biased.sample(n=240, random_state=10)
sampled_mgente_biased = mgente_biased.sample(n=1500, random_state=10)
sampled_lardelli_neutral = lardelli_neutral.sample(n=240, random_state=10)
# sampled_wiki_neutral = wiki_neutral.sample(n=200, random_state=10)
sampled_mgente_neutral = mgente_neutral.sample(n=1500, random_state=10)

# Combine, shuffle, save
final_df = pd.concat([
    sampled_lardelli_biased,
    sampled_mgente_biased,
    sampled_lardelli_neutral,
#    sampled_wiki_neutral,
    sampled_mgente_neutral
], ignore_index=True)

final_df = final_df.sample(frac=1, random_state=10).reset_index(drop=True)
final_df.to_csv("datasets/mgente_lardelli.csv", index=False)

print("Saved to datasets/")
