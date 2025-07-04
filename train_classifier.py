import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# 1. Load dataset
df = pd.read_csv("datasets/dataset.csv")

# 2. Dataset class
class BiasDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        english = self.data.iloc[idx]["english"]
        german = self.data.iloc[idx]["german"]
        label = int(self.data.iloc[idx]["label"])
        text_pair = english + " [SEP] " + german  # Treat as one input
        inputs = self.tokenizer(text_pair, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item["labels"] = torch.tensor(label)
        return item

# 3. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.to(device)

# 4. Dataloader
dataset = BiasDataset(df, tokenizer)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 5. Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# 6. Training loop (very basic, 3 epochs)
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
