import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_scheduler,
)
from sklearn.model_selection import train_test_split

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
        text_pair = english + " [SEP] " + german
        inputs = self.tokenizer(
            text_pair,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item["labels"] = torch.tensor(label)
        return item

# 3. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=2
)
model.to(device)

# 4. Split data
train_df, val_df = train_test_split(df, test_size=0.1, random_state=10)
train_dataset = BiasDataset(train_df, tokenizer)
val_dataset = BiasDataset(val_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# 5. Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_training_steps = len(train_loader) * 3
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 6. Training loop
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}")

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
    print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")
    model.train()

# Save model and tokenizer
output_dir = "./model_output"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")